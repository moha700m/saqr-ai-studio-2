"""SAQR command line.

    saqr probe                       hardware report + verified backend
    saqr models list                 installed / missing models and licences
    saqr models download [--all]     download models (asks to accept each licence)
    saqr swap -s SRC -t TARGET -o OUT [--preset balanced] [--backend cpu] [--no-watermark]
    saqr selftest                    load every model and run one inference
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _read_image(path: str):
    import cv2
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)  # works with non-ASCII Windows paths
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image: {path}")
    return img


def _setup(backend: str | None):
    from .config import SettingsStore
    from .hardware import detect_hardware
    from .models.manager import ModelManager
    from .runtime.providers import select_backend
    from .runtime.session import STATE

    hw = detect_hardware()
    sel = select_backend(hw, forced=backend or None)
    STATE.selection = sel
    store = SettingsStore()
    return hw, sel, ModelManager(settings=store), store


def cmd_probe(a) -> int:
    hw, sel, mm, _ = _setup(a.backend)
    out = {"hardware": hw.to_dict(), "backend": sel.to_dict(),
           "recommended_swapper": mm.recommended_swapper(sel.selected)}
    print(json.dumps(out, indent=2))
    print(f"\nSelected backend: {sel.label}" + (f"  ({sel.fallback_reason})" if sel.fallback_reason else ""),
          file=sys.stderr)
    return 0


def cmd_models(a) -> int:
    _, sel, mm, _ = _setup(a.backend)
    if a.action == "list":
        for st in mm.list():
            mark = "OK " if st.installed else "-- "
            print(f"{mark}{st.spec.id:22s} {st.spec.size / 1e6:8.1f} MB  {st.spec.role:10s} {st.spec.license}")
        miss = mm.missing_required()
        print("\nMissing:", ", ".join(miss) if miss else "nothing")
        return 0
    wanted = [m for m in mm.specs.values()
              if a.all or m.required or m.id == mm.recommended_swapper(sel.selected) or m.id in (a.ids or [])]
    for spec in wanted:
        if mm.path_for(spec.id):
            print(f"already installed: {spec.id}")
            continue
        if not mm.license_accepted(spec.id):
            print(f"\n{spec.title}\nLicence: {spec.license}\n{spec.license_url}")
            if a.yes:
                print("(accepted via --yes)")
            elif input("Accept this licence and download? [y/N] ").strip().lower() not in ("y", "yes"):
                print("skipped")
                continue
            mm.accept_license([spec.id])
        t0 = time.time()

        def prog(mid, done, total, _last=[0.0]):  # noqa: B006
            if time.time() - _last[0] > 0.5 or done == total:
                _last[0] = time.time()
                print(f"\r  {mid}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

        mm.download(spec.id, progress=prog)
        print(f"\n  verified SHA-256, {time.time() - t0:.0f}s")
    return 0


def cmd_swap(a) -> int:
    from .pipeline.config import preset
    from .pipeline.engine import FaceEngine
    from .safety.consent import CONSENT_TEXT, attest
    from .safety.watermark import draw_badge, write_image

    _, sel, mm, _ = _setup(a.backend)
    miss = mm.missing_required()
    if miss:
        raise SystemExit("missing models: " + ", ".join(miss) + "\nrun: saqr models download")
    if not a.i_have_consent:
        print(CONSENT_TEXT)
        if input("Type 'yes' to confirm: ").strip().lower() != "yes":
            raise SystemExit("consent not confirmed")
    eng = FaceEngine(mm, sel, preset(a.preset))
    if a.many:
        eng.cfg.update(face_selection="all")
    if a.mouth_mask:
        eng.cfg.update(mouth_mask=a.mouth_mask)
    if a.enhance:
        eng.cfg.update(enhance=True)
    eng.add_source(_read_image(a.source), attest("permission"))
    target = _read_image(a.target)
    t0 = time.perf_counter()
    eng.swapper, eng.landmarker  # noqa: B018 - load models before timing
    if eng.cfg.enhance:
        eng.enhancer  # noqa: B018
    load_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    res = eng.process(target, force_detect=True)
    dt = (time.perf_counter() - t0) * 1000
    if res.swapped == 0:
        raise SystemExit("no face found in the target image")
    if eng.check_nsfw(res.frame) >= 0.8:
        raise SystemExit("output blocked by the NSFW safety filter")
    out = res.frame
    if not a.no_watermark:
        draw_badge(out)
    write_image(Path(a.output), out, metadata=not a.no_metadata)
    print(f"saved {a.output}  ({res.swapped} face(s), processing {dt:.0f} ms, model load {load_ms:.0f} ms, "
          f"backend {eng.backends().get('swapper')}, model {eng.swapper_id})")
    return 0


def cmd_selftest(a) -> int:
    _, sel, mm, _ = _setup(a.backend)
    ok = True
    print(f"backend: {sel.label}")
    for st in mm.list():
        if not st.installed:
            print(f"  {st.spec.id:22s} not installed")
            continue
        r = mm.test(st.spec.id, sel)
        ok &= r.get("ok", False)
        print(f"  {st.spec.id:22s} {json.dumps(r)}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="saqr", description="SAQR AI Studio engine")
    p.add_argument("--backend", default="", help="force a backend: cuda, tensorrt, directml, openvino, cpu ...")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe").set_defaults(fn=cmd_probe)
    m = sub.add_parser("models")
    m.add_argument("action", choices=["list", "download"])
    m.add_argument("ids", nargs="*")
    m.add_argument("--all", action="store_true")
    m.add_argument("--yes", action="store_true", help="accept model licences non-interactively")
    m.set_defaults(fn=cmd_models)
    s = sub.add_parser("swap")
    s.add_argument("-s", "--source", required=True)
    s.add_argument("-t", "--target", required=True)
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--preset", default="balanced", choices=["fast", "balanced", "quality"])
    s.add_argument("--many", action="store_true", help="swap every face in the target")
    s.add_argument("--enhance", action="store_true")
    s.add_argument("--mouth-mask", type=float, default=0.0)
    s.add_argument("--no-watermark", action="store_true", help="exports only")
    s.add_argument("--no-metadata", action="store_true", help="exports only")
    s.add_argument("--i-have-consent", action="store_true")
    s.set_defaults(fn=cmd_swap)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

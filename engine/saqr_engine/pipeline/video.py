"""Video processing through FFmpeg pipes (no intermediate frame dumps).

    ffmpeg -i input -> rawvideo bgr24 pipe -> FaceEngine -> rawvideo pipe -> ffmpeg encoder (+ original audio)

Supports progress callbacks, cancellation, trimming, output resolution and
quality (CRF), hardware encoders with automatic software fallback, and always
cleans its temporary file up.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .. import config
from ..safety.watermark import METADATA_TEXT, draw_badge

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class VideoCancelled(RuntimeError):
    pass


def ffmpeg_path(name: str = "ffmpeg") -> Optional[str]:
    exe = name + (".exe" if sys.platform == "win32" else "")
    env = os.environ.get("SAQR_FFMPEG_DIR")
    cands = []
    if env:
        cands.append(Path(env) / exe)
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
        cands += [base / "ffmpeg" / exe, base / exe]
    for c in cands:
        if c.is_file():
            return str(c)
    return shutil.which(name)


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    has_audio: bool
    codec: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def probe(path: str) -> VideoInfo:
    ffprobe = ffmpeg_path("ffprobe")
    if not ffprobe:
        raise FileNotFoundError("ffprobe not found")
    out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                         capture_output=True, text=True, creationflags=_NO_WINDOW, check=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    dur = float(v.get("duration") or d.get("format", {}).get("duration") or 0)
    frames = int(v.get("nb_frames") or 0) or int(round(dur * fps))
    return VideoInfo(int(v["width"]), int(v["height"]), round(fps, 3), frames, dur,
                     any(s["codec_type"] == "audio" for s in d["streams"]), v.get("codec_name", "?"))


def read_frame_at(path: str, t: float, max_w: int = 1280) -> Optional[np.ndarray]:
    """Grab one frame (for the timeline preview)."""
    import cv2

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if frame.shape[1] > max_w:
        s = max_w / frame.shape[1]
        frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return frame


def _encoders(ffmpeg: str) -> set[str]:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True,
                             creationflags=_NO_WINDOW, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {ln.split()[1] for ln in out.splitlines() if ln.startswith(" V")}


_ENCODER_OK: dict[str, bool] = {}


def _working_encoder(ffmpeg: str, order: list[str]) -> str:
    """First encoder that really encodes a test frame (hardware encoders are listed even without the GPU)."""
    for e in order:
        if e == "libx264":
            return e
        if e not in _ENCODER_OK:
            r = subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                                "-frames:v", "2", "-c:v", e, "-f", "null", "-"], capture_output=True,
                               creationflags=_NO_WINDOW, timeout=20)
            _ENCODER_OK[e] = r.returncode == 0
        if _ENCODER_OK[e]:
            return e
    return "libx264"


@dataclass
class VideoJob:
    input: str
    output: str
    start: float = 0.0
    end: float = 0.0  # 0 = to the end
    scale: float = 1.0  # output resolution factor
    crf: int = 20  # 0..51, lower = better
    encoder: str = "auto"  # auto | libx264 | h264_nvenc | h264_amf | h264_qsv
    keep_audio: bool = True
    watermark: bool = True
    metadata: bool = True
    nsfw_every: int = 30  # frames


def process_video(engine, job: VideoJob, progress: Optional[Callable[[int, int, float], None]] = None,
                  cancel: Optional[threading.Event] = None, preview: Optional[Callable[[np.ndarray], None]] = None) -> dict:
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise FileNotFoundError("FFmpeg not found")
    info = probe(job.input)
    w, h = info.width, info.height
    ow, oh = int(w * job.scale) // 2 * 2, int(h * job.scale) // 2 * 2
    t0, t1 = max(0.0, job.start), (job.end if job.end > 0 else info.duration)
    total = max(1, int(round((t1 - t0) * info.fps))) if info.duration else info.frames
    tmp = config.temp_dir() / f"saqr_{os.getpid()}_{int(time.time() * 1000)}.mp4"

    dec_cmd = [ffmpeg, "-v", "error", "-ss", f"{t0:.3f}"]
    if job.end > 0:
        dec_cmd += ["-to", f"{t1:.3f}"]
    dec_cmd += ["-i", job.input, "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-"]

    encs = _encoders(ffmpeg)
    order = [job.encoder] if job.encoder != "auto" else ["h264_nvenc", "h264_amf", "h264_qsv", "libx264"]
    order = [e for e in order if e in encs] or ["libx264"]

    def enc_cmd(encoder: str) -> list[str]:
        c = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
             "-r", f"{info.fps}", "-i", "-"]
        if encoder == "libx264":
            c += ["-c:v", "libx264", "-preset", "medium", "-crf", str(job.crf)]
        elif encoder == "h264_nvenc":
            c += ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(job.crf)]
        elif encoder == "h264_amf":
            c += ["-c:v", "h264_amf", "-quality", "quality", "-qp_i", str(job.crf), "-qp_p", str(job.crf)]
        else:
            c += ["-c:v", encoder, "-global_quality", str(job.crf)]
        c += ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if job.metadata:
            c += ["-metadata", f"comment={METADATA_TEXT}", "-metadata", "encoder=SAQR AI Studio"]
        c.append(str(tmp))
        return c

    import cv2

    started = time.perf_counter()
    done = swapped = blocked = 0
    used_encoder = None
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
    enc = None
    try:
        frame_bytes = w * h * 3
        engine.reset_tracking()
        block_until = -1
        while True:
            if cancel is not None and cancel.is_set():
                raise VideoCancelled("cancelled")
            buf = dec.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            res = engine.process(frame)
            out = res.frame
            if res.swapped:
                swapped += 1
                if done % max(1, job.nsfw_every) == 0 and engine.check_nsfw(out) >= 0.8:
                    block_until = done + job.nsfw_every
                if done <= block_until:
                    out = cv2.GaussianBlur(out, (0, 0), 30)
                    blocked += 1
            if (ow, oh) != (w, h):
                out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_AREA if job.scale < 1 else cv2.INTER_CUBIC)
            if job.watermark:
                draw_badge(out)
            if enc is None:
                used_encoder = _working_encoder(ffmpeg, order)
                enc = subprocess.Popen(enc_cmd(used_encoder), stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                       creationflags=_NO_WINDOW)
            try:
                enc.stdin.write(np.ascontiguousarray(out).tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError(f"encoder {used_encoder} failed: {enc.stderr.read().decode(errors='replace')[-400:]}") from exc
            done += 1
            if progress and (done % 5 == 0 or done == total):
                progress(done, total, done / max(1e-6, time.perf_counter() - started))
            if preview and done % 15 == 0:
                preview(out)
        if enc is None:
            raise RuntimeError("no frames decoded: " + dec.stderr.read().decode(errors="replace")[-400:])
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError(f"encoder failed: {enc.stderr.read().decode(errors='replace')[-400:]}")
        dec.wait()
        final = Path(job.output)
        final.parent.mkdir(parents=True, exist_ok=True)
        if job.keep_audio and info.has_audio:
            mux = [ffmpeg, "-v", "error", "-y", "-i", str(tmp), "-ss", f"{t0:.3f}"]
            if job.end > 0:
                mux += ["-to", f"{t1:.3f}"]
            mux += ["-i", job.input, "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest"]
            if job.metadata:
                mux += ["-metadata", f"comment={METADATA_TEXT}"]
            mux.append(str(final))
            r = subprocess.run(mux, capture_output=True, creationflags=_NO_WINDOW)
            if r.returncode != 0:
                shutil.move(str(tmp), final)  # keep the video even if audio remux failed
        else:
            shutil.move(str(tmp), final)
        elapsed = time.perf_counter() - started
        return {"output": str(final), "frames": done, "swapped_frames": swapped, "blocked_frames": blocked,
                "seconds": round(elapsed, 2), "fps": round(done / elapsed, 2) if elapsed else 0,
                "encoder": used_encoder, "input": info.to_dict()}
    finally:
        for p in (dec, enc):
            if p is not None and p.poll() is None:
                p.kill()
        tmp.unlink(missing_ok=True)

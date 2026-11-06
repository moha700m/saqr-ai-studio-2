import json

from saqr_engine.hardware import detect_hardware
from saqr_engine.hardware.probe import (guess_integrated, merge_gpus, parse_lspci, parse_windows_gpus,
                                        probe_windows_gpus, vendor_from_name)
from saqr_engine.hardware.report import GPUInfo, HardwareReport

WIN_JSON = json.dumps({
    "vc": [
        {"Name": "NVIDIA GeForce RTX 4070", "AdapterRAM": 4293918720, "DriverVersion": "32.0.15.6094",
         "PNPDeviceID": "PCI\\VEN_10DE&DEV_2786", "AdapterCompatibility": "NVIDIA"},
        {"Name": "Intel(R) UHD Graphics 770", "AdapterRAM": 1073741824, "DriverVersion": "31.0.101.4502",
         "PNPDeviceID": "PCI\\VEN_8086&DEV_4680", "AdapterCompatibility": "Intel Corporation"},
    ],
    "reg": [{"Name": "NVIDIA GeForce RTX 4070", "Mem": 12878610432, "MemDw": None}],
})


def test_parse_windows_gpus_uses_registry_vram_beyond_4gb():
    gpus = parse_windows_gpus(WIN_JSON)
    assert [g.vendor for g in gpus] == ["nvidia", "intel"]
    assert gpus[0].vram_mb == 12282  # registry qwMemorySize, not the 4 GiB-capped WMI value
    assert gpus[0].integrated is False and gpus[1].integrated is True
    assert "cuda" in gpus[0].apis and "directml" in gpus[0].apis


def test_parse_windows_single_object_and_binary_mem():
    one = json.dumps({"vc": {"Name": "AMD Radeon RX 7800 XT", "AdapterRAM": 4293918720,
                             "PNPDeviceID": "PCI\\VEN_1002&DEV_747E", "DriverVersion": "31.0.24027.1012"},
                      "reg": {"Name": "AMD Radeon RX 7800 XT", "Mem": list((17163091968).to_bytes(8, "little"))}})
    g = parse_windows_gpus(one)[0]
    assert g.vendor == "amd" and g.vram_mb == 16368 and g.integrated is False


def test_probe_windows_handles_failure():
    errors = []
    assert probe_windows_gpus(errors, run=lambda c, t: None) == []
    assert errors


def test_parse_lspci():
    txt = ("00:02.0 VGA compatible controller [0300]: Intel Corporation Alder Lake-P GT2 [Iris Xe Graphics] [8086:46a6] (rev 0c)\n"
           "01:00.0 3D controller [0302]: NVIDIA Corporation AD107M [GeForce RTX 4060 Max-Q / Mobile] [10de:28a0] (rev a1)\n")
    gpus = parse_lspci(txt)
    assert [g.vendor for g in gpus] == ["intel", "nvidia"]
    assert "rev" not in gpus[0].name


def test_vendor_and_integrated_heuristics():
    assert vendor_from_name("Radeon(TM) Graphics") == "amd"
    assert guess_integrated("amd", "AMD Radeon(TM) Graphics") is True
    assert guess_integrated("intel", "Intel(R) Arc(TM) A770 Graphics") is False
    assert guess_integrated("intel", "Intel(R) Arc(TM) Graphics") is True


def test_merge_prefers_nvml_and_dedupes():
    nv = [GPUInfo(vendor="nvidia", name="NVIDIA GeForce RTX 4070", vram_mb=12282, driver_version="560.94",
                  integrated=False, source="nvml", apis=["cuda"])]
    merged = merge_gpus(nv, parse_windows_gpus(WIN_JSON))
    assert len(merged) == 2
    assert merged[0].source == "nvml" and merged[0].driver_version == "560.94"
    assert "directml" in merged[0].apis


def test_detect_real_machine_and_roundtrip():
    hw = detect_hardware()
    assert hw.cpu.logical_cores > 0 and hw.system.ram_total_mb > 0
    again = HardwareReport.from_dict(json.loads(hw.to_json()))
    assert again.cpu.model == hw.cpu.model

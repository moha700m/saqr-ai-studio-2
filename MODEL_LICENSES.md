# Model licences

SAQR AI Studio's **code** is licensed under AGPL-3.0. SAQR does **not** ship any model weights. The installer and the
Model Manager download them from their public sources **after you accept the licences**, and check every file
against the SHA-256 checksum listed in `engine/saqr_engine/models/registry.json`.

| Model | File | Used for | Licence | Commercial use |
|---|---|---|---|---|
| SCRFD 10G (buffalo_l) | `det_10g.onnx` | face detection | InsightFace: non-commercial research only | **No** |
| ArcFace R50 (buffalo_l) | `w600k_r50.onnx` | identity embedding | InsightFace: non-commercial research only | **No** |
| 2D-106 landmarks (buffalo_l) | `2d106det.onnx` | face shape / masks | InsightFace: non-commercial research only | **No** |
| inswapper 128 (FP32 / FP16) | `inswapper_128*.onnx` | face swap | InsightFace: non-commercial research only; commercial licence from InsightFace | **No** |
| GFPGAN v1.4 | `GFPGANv1.4.onnx` | face enhancement (optional) | Apache-2.0 (TencentARC); trained on FFHQ | Yes (check FFHQ terms) |
| NSFW ViT (int8) | `nsfw_vit_int8.onnx` | always-on safety filter | Apache-2.0 (Falconsai/nsfw_image_detection) | Yes |

Sources:
- InsightFace licence: https://github.com/deepinsight/insightface#license. It says "the training data … and the models
  trained with these data are available for non-commercial research purposes only". The inswapper and buffalo_l
  models need a separate licence from InsightFace for commercial use.
- Weights mirror: https://huggingface.co/hacksider/deep-live-cam. The repository's "gpl-3.0" tag does **not** change
  the original model licences above.
- GFPGAN: https://github.com/TencentARC/GFPGAN/blob/master/LICENSE
- NSFW classifier: https://huggingface.co/Falconsai/nsfw_image_detection and the ONNX export
  https://huggingface.co/onnx-community/nsfw_image_detection-ONNX

**What this means:** SAQR AI Studio, as distributed with these models, is for **personal and non-commercial research
use**. A commercial product would need a licence from InsightFace or a different, commercially licensed swap model.

Test fixtures in `tests/fixtures/` are CC0 museum paintings from Wikimedia Commons (see `tests/fixtures/LICENSES.md`).

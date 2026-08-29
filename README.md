# AI-Based Fake Identity & Document Screening System

**SIH 2026 · Problem Statement 26188 · Ministry of Home Affairs (Sashastra Seema Bal)**

An end-to-end document screening platform that analyzes identity/travel documents,
detects tampering & forged stamps, runs image-metadata forensics, extracts identity
fields (OCR), validates them against official formats/rules, and verifies the
document owner against a live selfie — then fuses everything into a standardized
risk score and verdict with a digital audit trail.

> **Zero training.** All components are pre-trained models (Hugging Face / canonical
> checkpoints). Nothing is trained from scratch unless explicitly requested.

## Modules (maps to SIH-26188)

| # | Module | Implementation | Pretrained model / method |
|---|--------|----------------|---------------------------|
| 1 | OCR Extraction | PAN / Aadhaar / passport-style fields (name, DOB, ID number, expiry, gender, signature) + raw OCR lines | PaddleOCR PP-OCRv6 (`paddleocr`) |
| 2 | Document Validation | format rules (PAN/DOB/Aadhaar/MRZ check-digits), expiry check, blacklist lookup, doc-type detection | rule engine + `passport-mrz`-style parser (self-contained) |
| 3 | Tampering Detection | document-level real/forged classification | `zodumair/document-forgery-detector` (ViT-B/16, ELA-blended) |
| 3a | Stamp Forgery | ink-color segmentation + circularity localization, ELA + edge forensics per region | OpenCV + Error Level Analysis |
| 3b | Image Metadata | EXIF/editing-software/date/GPS/threshold forensics | `exifread` |
| 4 | Face Verification | document-photo vs selfie similarity (VAggFace) + morph heuristics | `facenet-pytorch` (MTCNN + InceptionResnetV1 VGGFace2) |
| — | Multi-identity linkage | same face / different name across screenings → flagged | local embedding watchlist |
| — | Risk fusion + trail | weighted fusion, strong-signal override, standardized verdict, JSON audit trail | `modules/risk.py` |

Second-opinion forgery model: `salmanzaman777/digital-image-forgery-detection-model`
(dual RGB+ELA CNN, CASIA v2) — runs on CPU to keep the GPU for torch + paddle.

## Dataset (bundled, CC BY 4.0 / research)

- `data/morphed/` — 500 Aadhaar-card images with faceswapper.ai-morphed faces (Mendeley `k7srr5wwhv`)
- `data/aadhar_labeled_imgs/` + `data/aadhar_labels*/` — 31 identity documents (Label Studio, YOLO boxes: name/dob/pan_number/signature)

## Quickstart

```bash
pip install -r requirements.txt
python app.py                     # Gradio UI at http://127.0.0.1:7860
python scripts/validate_morph.py  # expect ~85-95% forged-detection on the morph set
python scripts/validate_fields.py # OCR vs Label-Studio GT boxes (IoU + extraction rate)
```

First run downloads the pretrained models (~500 MB total) into the HF/paddle caches.

## Layout

```
app.py                     Gradio UI
config.py                  thresholds, risk weights, model registry
modules/
  forge_detector.py        Module 3 – document forgery (ViT + ELA CNN)
  stamp_forensics.py       Module 3a – stamp & region forensics
  metadata_analysis.py     Module 3b – EXIF / metadata
  ocr_fields.py            Module 1 – OCR extraction + regex fields
  document_validation.py   Module 2 – format rules, expiry, blacklist
  face_verify.py           Module 4 – face verification
  watchlist.py             multi-identity linkage
  models.py                lazy pretrained-model loaders
  risk.py                  fusion, verdict, digital trail
scripts/valid*.py          dataset validation harnesses
data/                      zishan datasets (morphed + labeled docs)
```

> PaddleOCR needs `FLAGS_enable_pir_api=0` + `enable_mkldnn=False` on paddle 3.x —
> handled automatically in `config.py`.
"""Central configuration: model registry, risk weights, thresholds, paths.

Standardized decisions come from thresholds here (single source of truth),
so checkpoints across an org produce identical verdicts.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIL_DIR = os.path.join(BASE_DIR, "trail")
MORPH_DIR = os.path.join(DATA_DIR, "morphed")
LABELED_IMG_DIR = os.path.join(DATA_DIR, "aadhar_labeled_imgs")
LABELED_GT_DIR = os.path.join(DATA_DIR, "aadhar_labels")
LABELED_GT_HR_DIR = os.path.join(DATA_DIR, "aadhar_labels_hr")

os.makedirs(TRAIL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# HF model registry (all PRETRAINED - nothing is trained from scratch)
# ---------------------------------------------------------------------------
MODELS = {
    # Document-level real/forged classifier (ViT-B/16 fine-tuned on
    # ELA-blended forged documents incl. stamp-overlay, copy-move, splicing).
    "forge_vit": {
        "repo": "zodumair/document-forgery-detector",
        "kind": "transformers",
    },
    # Dual-branch RGB + ELA CNN (CASIA v2), used as a second opinion.
    "forge_ela_cnn": {
        "repo": "salmanzaman777/digital-image-forgery-detection-model",
        "file": "M3_best_v2.h5",
        "kind": "keras",
        "enabled": bool(os.environ.get("FORGE_KERAS", "1") == "1"),
    },
    # Face detection + embeddings (VGGFace2). Not HF-hosted but pretrained.
    "face": {"backend": "facenet", "embedder": "InceptionResnetV1(vggface2)"},
    "ocr": {"repo": "paddleocr PP-OCRv6_medium_det/rec (paddlex)", "lang": "en"},
}

# ELA preprocessing constants - must match zodumair training recipe
ELA_JPEG_QUALITY = 90
ELA_SCALE = 15
ELA_FINAL_JPEG_QUALITY = 75
ELA_BLEND_ALPHA = 0.3  # forged detector mixes original + ELA residual at 0.3

# ---------------------------------------------------------------------------
# Risk fusion weights (sum = 1.0)
# ---------------------------------------------------------------------------
RISK_WEIGHTS = {
    "forgery": 0.25,
    "stamp": 0.15,
    "metadata": 0.10,
    "face": 0.20,
    "ocr": 0.15,
    "validation": 0.15,
}

# A single high-confidence signal forces at least a manual review.
STRONG_SIGNAL_OVERRIDE = {
    "forgery": 0.85,  # forge prob above this => cannot be auto-PASS
    "stamp": 0.85,
    "metadata": 0.80,
    "face": 0.80,
    "ocr": 0.80,
    "validation": 0.85,
}

# ---------------------------------------------------------------------------
# Standardized verdict thresholds
# ---------------------------------------------------------------------------
VERDICT = {
    "pass": 0.34,
    "review": 0.66,  # above => reject; between pass and review => manual review
}

# ---------------------------------------------------------------------------
# Per-module thresholds
# ---------------------------------------------------------------------------
FACE_MATCH_THRESHOLD = 0.50  # cosine similarity (facenet space)
STAMP_MIN_CIRCULARITY = 0.55
STAMP_MIN_RADIUS_RATIO = 0.02  # of min(image w,h)
PASS_INDICATOR_TEXT = ("signature", "sign", "stamp", "seal")

# ---------------------------------------------------------------------------
# PaddleOCR hardening (paddle 3.x PIR/mkldnn bug workaround)
# ---------------------------------------------------------------------------
os.environ.setdefault("FLAGS_enable_pir_api", "0")
try:
    import paddle

    paddle.set_flags({"FLAGS_enable_pir_api": 0})
except Exception:
    pass

OCR_KWARGS = dict(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
    enable_mkldnn=False,
)

# ---------------------------------------------------------------------------
# Metadata forensics heuristics
# ---------------------------------------------------------------------------
SUSPICIOUS_SOFTWARE = (
    "photoshop",
    "paint.net",
    "gimp",
    "adobe illustrator",
    "photoscape",
    "pixlr",
    "snapseed",
    "camscanner pro",
    "adobe acrobat pro",
    "picsart",
)
SCREENSHOT_SOFTWARE = ("snipping", "screen", "capture", "bandicam", "psd")
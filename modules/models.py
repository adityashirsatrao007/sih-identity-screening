"""Shared pretrained-model loaders (lazy singletons) + ELA helpers."""

from __future__ import annotations

import io
import os
import threading

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

_lock = threading.Lock()
_cache: dict = {}

DEVICE = (
    "cpu"
    if os.environ.get("FORCE_CPU")
    else "cuda" if __import__("torch").cuda.is_available() else "cpu"
)


def elt_image(pil_img: Image.Image, quality: int = 90, scale: int = 15) -> Image.Image:
    """Error Level Analysis residual. Re-encode at `quality`, diff with original.

    High residual in a region = inconsistent re-compression = tampering hint.
    """
    pil_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    diff = ImageChops.difference(pil_img, Image.open(buf).convert("RGB"))
    return ImageEnhance.Brightness(diff).enhance(scale)


def ela_blend(pil_img: Image.Image, alpha: float = 0.3) -> Image.Image:
    """Original blended with its ELA residual (matches zodumair prep)."""
    orig = pil_img.convert("RGB")
    ela = elt_image(orig)
    return Image.blend(orig, ela, alpha)


def ela_residual_stats(pil_img: Image.Image, quality: int = 90):
    """Numerical ELA statistics over a region (mean/std of amplified residual)."""
    arr = np.asarray(elt_image(pil_img, quality=quality).convert("L"), dtype=np.float32)
    return {"ela_mean": float(arr.mean()), "ela_std": float(arr.std()), "ela_p95": float(np.percentile(arr, 95))}


def get_forge_vit():
    """Load zodumair/document-forgery-detector (ViT-B/16, transformers)."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    with _lock:
        if "forge_vit" not in _cache:
            model = AutoModelForImageClassification.from_pretrained("zodumair/document-forgery-detector")
            proc = AutoImageProcessor.from_pretrained("zodumair/document-forgery-detector")
            if DEVICE == "cuda":
                model = model.to("cuda")
            model.eval()
            _cache["forge_vit"] = (model, proc)
    return _cache["forge_vit"]


def get_forge_ela_cnn():
    """Load salmanzaman777 dual-branch RGB+ELA CNN (Keras). Optional.

    Runs on CPU on purpose: sharing the 4GB GPU with torch + paddle causes
    CUDA OOM / TF-XLA aborts in the same process.
    """
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    from huggingface_hub import hf_hub_download

    with _lock:
        if "forge_ela_cnn" not in _cache:
            path = hf_hub_download(
                repo_id="salmanzaman777/digital-image-forgery-detection-model",
                filename="M3_best_v2.h5",
            )
            _cache["forge_ela_cnn"] = tf.keras.models.load_model(path, compile=False)
    return _cache["forge_ela_cnn"]


def get_face_pipeline():
    """facenet-pytorch MTCNN detector + InceptionResnetV1 (VGGFace2) embedder."""
    from facenet_pytorch import MTCNN, InceptionResnetV1

    with _lock:
        if "face" not in _cache:
            mtcnn = MTCNN(keep_all=False, device=DEVICE, min_face_size=12)
            res = InceptionResnetV1(pretrained="vggface2").eval()
            if DEVICE == "cuda":
                res = res.to("cuda")
            _cache["face"] = (mtcnn, res)
    return _cache["face"]


def get_ocr():
    from paddleocr import PaddleOCR

    with _lock:
        if "ocr" not in _cache:
            from config import OCR_KWARGS

            _cache["ocr"] = PaddleOCR(**OCR_KWARGS)
    return _cache["ocr"]
"""Module 1 - Document forgery / tampering detection (pretrained ViT + ELA CNN)."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from PIL import Image

from modules.models import DEVICE, ela_blend, get_forge_ela_cnn, get_forge_vit

warnings.filterwarnings("ignore")

_IMG = (224, 224)


def _vit_predict(pil_img: Image.Image):
    """ELA-blended ViT (zodumair). Returns forged probability in [0,1]."""
    model, proc = get_forge_vit()
    blended = ela_blend(pil_img)
    inp = proc(images=[blended], return_tensors="pt")
    inp = {k: (v.to(DEVICE) if DEVICE == "cuda" else v) for k, v in inp.items()}
    with torch.no_grad():
        logits = model(**inp).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    # label order: {0: 'real', 1: 'forged'}
    forged = float(probs[1]) if probs.shape[0] > 1 else float(probs[0])
    return forged


def _ela_cnn_predict(pil_img: Image.Image):
    """Dual RGB+ELA CNN (salmanzaman, CASIA v2) second opinion."""
    model = get_forge_ela_cnn()
    import tensorflow as tf

    def _ela(im: Image.Image, q=90, scale=15):
        import io

        from PIL import ImageChops, ImageEnhance

        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        buf.seek(0)
        d = ImageEnhance.Brightness(ImageChops.difference(im, Image.open(buf).convert("RGB"))).enhance(scale)
        out = io.BytesIO()
        d.save(out, "JPEG", quality=75)
        x = tf.image.resize(tf.image.decode_jpeg(out.getvalue(), channels=3), _IMG)
        return (tf.cast(x, tf.float32) / 255.0).numpy()

    rgb = np.array(pil_img.resize(_IMG, Image.LANCZOS), np.float32)[np.newaxis] / 255.0
    pred = float(model.predict([rgb, _ela(pil_img)[np.newaxis]], verbose=0)[0][0])
    return pred


def analyze_document(pil_img: Image.Image) -> dict:
    """Run document-level forgery screening. Returns structured result."""

    def _to_pil(img):
        return img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")

    img = _to_pil(pil_img)
    forge_prob = _vit_predict(img)

    ela_cnn_prob = None
    try:
        from config import MODELS

        if MODELS["forge_ela_cnn"]["enabled"]:
            ela_cnn_prob = float(_ela_cnn_predict(img))
    except Exception:
        ela_cnn_prob = None

    combined = forge_prob
    if ela_cnn_prob is not None:
        combined = 0.6 * forge_prob + 0.4 * ela_cnn_prob
    # Strong-signal boost: a confident ViT call must not be averaged into silence.
    if forge_prob >= 0.90:
        combined = max(combined, 0.85)
    elif forge_prob <= 0.25:
        combined = min(combined, 0.35)
    combined = float(np.clip(combined, 0.0, 1.0))

    risk = float(np.clip((combined - 0.5) * 2.0, 0.0, 1.0))  # 0..1 scale
    verdict = "FORGED" if combined >= 0.5 else "AUTHENTIC"
    return {
        "forge_probability": round(combined, 4),
        "verdict": verdict,
        "risk": round(risk, 4),
        "models": {
            "vit_ela_blend": round(forge_prob, 4),
            "ela_cnn_casia2": round(ela_cnn_prob, 4) if ela_cnn_prob is not None else None,
        },
    }
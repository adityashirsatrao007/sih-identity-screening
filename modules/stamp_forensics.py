"""Module - Stamp forgery / region forensics.

Locates candidate stamp/seal/logo regions by colored-ink segmentation
(red / blue / violet) + circularity, then runs ELA forensics + edge analysis
on each region. A forged or cloned stamp shows abnormal compression
residual / re-sampled boundaries vs. the printed background.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from config import STAMP_MIN_CIRCULARITY, STAMP_MIN_RADIUS_RATIO
from modules.models import ela_residual_stats

_INK_RANGES = {
    "red": ((0, 90, 40), (10, 255, 255)),
    "red2": ((170, 90, 40), (180, 255, 255)),
    "blue": ((100, 60, 40), (130, 255, 255)),
    "violet": ((130, 60, 40), (165, 255, 255)),
}


def _stamp_regions(pil_img: Image.Image, max_candidates: int = 6):
    """Return list of dicts with bbox pixels + circularity for ink-colored blobs."""
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    min_r = max(2, int(STAMP_MIN_RADIUS_RATIO * min(h, w)))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros((h, w), np.uint8)
    for lo, hi in _INK_RANGES.values():
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < (np.pi * min_r * min_r):
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < min_r:
            continue
        circularity = 4 * np.pi * area / max(1e-6, (cv2.arcLength(c, True) ** 2))
        x, y, cw, ch = cv2.boundingRect(c)
        regions.append(
            {"bbox": (x, y, cw, ch), "center": (cx, cy), "radius": radius,
             "circularity": float(circularity), "area": float(area), "ink_circle": float(circularity)}
        )
    regions = [r for r in regions if r["circularity"] >= STAMP_MIN_CIRCULARITY]
    regions.sort(key=lambda r: -r["area"])
    return regions[:max_candidates]


def _region_ela_score(pil_img: Image.Image, bbox: tuple) -> dict:
    x, y, cw, ch = bbox
    crop = pil_img.crop((x, y, x + cw, y + ch)) if cw > 4 and ch > 4 else pil_img
    stats = ela_residual_stats(crop, quality=90)
    gr = np.asarray(crop.convert("L"), dtype=np.float32)
    edge_ratio = float(np.mean(cv2.Canny(np.uint8(gr), 60, 140) > 0))
    stats["edge_ratio"] = round(edge_ratio, 4)
    return stats


def analyze_stamps(pil_img: Image.Image) -> dict:
    """Detect official-ink regions and triage them for forgery features."""
    img = pil_img.convert("RGB") if isinstance(pil_img, Image.Image) else Image.open(pil_img).convert("RGB")
    regions = _stamp_regions(img)
    results = []
    for r in regions:
        stats = _region_ela_score(img, r["bbox"])
        # Strong/localized ELA + low edge density = suspicious re-print area.
        local_score = float(np.clip((stats["ela_mean"] - 9.0) / 30.0, 0.0, 1.0))
        results.append(
            {"bbox": r["bbox"], "circularity": round(r["circularity"], 3), "area": int(r["area"]),
             **stats, "suspicious_score": round(local_score, 4)}
        )
    if results:
        risk = float(np.clip(max(r["suspicious_score"] for r in results) - 0.1, 0.0, 1.0))
    else:
        risk = 0.0  # no stamp/ink present - not evidence of forgery on its own
    return {
        "stamp_count": len(results),
        "stamps": results,
        "risk": round(risk, 4),
        "note": ("No official-ink region found. Not counted as a forgery signal."
                 if not results else "Stamp/seal regions localized and foraged for anomalies."),
    }
"""Module 4 - Face verification (doc owner vs. presented person) + morph/heuristics.

Uses pretrained MTCNN (face detection) + InceptionResnetV1 (VGGFace2) embeddings
(facenet-pytorch). No training.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from config import FACE_MATCH_THRESHOLD
from modules.models import DEVICE, ela_residual_stats, get_face_pipeline


def _rgb(img) -> Image.Image:
    return img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")


def _embed_face(pil_img: Image.Image):
    """Detect + embed the largest face. Returns (embedding, bbox, conf) or None."""
    mtcnn, resnet = get_face_pipeline()
    b = pil_img.copy()
    try:
        boxes, conf = mtcnn.detect(b)
        if boxes is None or len(boxes) == 0:
            return None
        idx = int(np.argmax(conf))
        box = boxes[idx]
        aligned, prob = mtcnn(b, return_prob=True)
        if aligned is None:
            return None
        emb = resnet(aligned.unsqueeze(0).to(DEVICE))
        return emb[0].detach().cpu().numpy(), [float(v) for v in box], float(prob)
    except Exception:
        return None


def face_verify(document_img, selfie_img) -> dict:
    """Compare the photo on a document with a selfie of the presenter."""
    doc = _rgb(document_img)
    face_doc = _embed_face(doc)
    face_selfie = None
    if selfie_img is not None:
        face_selfie = _embed_face(_rgb(selfie_img))

    sim = None
    match = None
    risk = 0.0
    findings = []

    if face_doc is None and face_selfie is None:
        # No reference face available on either side — cannot verify.
        risk, match, findings = 0.35, "INCONCLUSIVE (no face detected)", ["No face detected on document or selfie."]
    elif face_doc is None:
        findings.append("No face detected on document image.")
        risk, match = 0.8, "REJECT (document has no verifiable face)"
    elif face_selfie is None:
        findings.append("No face detected in selfie.")
        risk, match = 0.75, "REJECT (no selfie face)"
    else:
        emb_doc, bbox_doc, conf_doc = face_doc
        emb_selfie, bbox_selfie, conf_selfie = face_selfie
        emb_doc = emb_doc / (np.linalg.norm(emb_doc) + 1e-9)
        emb_selfie = emb_selfie / (np.linalg.norm(emb_selfie) + 1e-9)
        sim = float(np.dot(emb_doc, emb_selfie))
        match = bool(sim >= FACE_MATCH_THRESHOLD)
        risk = float(np.clip(1.0 - (sim - 0.2) / 0.8, 0.0, 1.0)) if match else 1.0
        findings.append(f"Cosine similarity {sim:.3f} vs threshold {FACE_MATCH_THRESHOLD}")

        # Document face morph / quality heuristics
        w, h = doc.size
        fw, fh = bbox_doc[2] - bbox_doc[0], bbox_doc[3] - bbox_doc[1]
        face_frac = (fw * fh) / (w * h)
        if face_frac < 0.02:
            findings.append(f"Document face is very small ({face_frac * 100.0:.1f}% of image) - possible low-grade crop")
        stats = ela_residual_stats(doc.crop([int(v) for v in bbox_doc]))
        if stats["ela_mean"] > 25:
            findings.append(f"High ELA residual in document face region ({stats['ela_mean']:.1f}) - possible morph/swap")

    return {
        "document_face_detected": face_doc is not None,
        "selfie_face_detected": face_selfie is not None,
        "doc_face_bbox": [int(v) for v in face_doc[1]] if face_doc else None,
        "selfie_face_bbox": [int(v) for v in face_selfie[1]] if face_selfie else None,
        "doc_embedding": (face_doc[0].tolist() if face_doc else None),
        "selfie_embedding": (face_selfie[0].tolist() if face_selfie else None),
        "similarity": round(sim, 4) if sim is not None else None,
        "match": match,
        "verdict": match if match is not None else (match if face_doc is not None and face_selfie is not None else None),
        "risk": round(risk, 4),
        "findings": findings,
    }
"""Gradio web UI - AI-Based Fake Identity & Document Screening System.

Modules: stamp & document forgery detection, metadata analysis, OCR field
extraction, face verification. Uses PRETRAINED models only.

Run:  python app.py  (then open the printed URL, default http://127.0.0.1:7860)
"""

from __future__ import annotations

import os
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("FORGE_KERAS", "1")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from config import MORPH_DIR, MODELS
from modules import (document_validation, forge_detector, metadata_analysis,
                     ocr_fields, risk, stamp_forensics, watchlist)

try:
    from modules.face_verify import face_verify
except ImportError:
    face_verify = None


def _annotate(img: Image.Image, stamps: list, fields_box: list) -> Image.Image:
    img = img.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    for s in stamps:
        x, y, cw, ch = s["bbox"]
        d.rectangle([x, y, x + cw, y + ch], outline=(0, 180, 0), width=3)
    if fields_box:
        for b in fields_box:
            if isinstance(b, list) and len(b) == 4:
                d.rectangle(b, outline=(0, 120, 255), width=2)
    return img


def _draw_face(img: Image.Image, bbox):
    if bbox:
        ImageDraw.Draw(img).rectangle([int(v) for v in bbox], outline=(255, 200, 0), width=3)
    return img


def _summary(ocr: dict) -> dict:
    f = ocr["fields"]
    return {
        "Document type": f.get("doc_type"),
        "Name": f.get("name"),
        "DOB": f.get("dob"),
        "PAN": f.get("pan_number"),
        "Aadhaar": f.get("aadhaar_number"),
        "Expiry": f.get("expiry"),
        "Gender": f.get("gender"),
        "Signature detected": f.get("has_signature"),
        "OCR lines": len(ocr["lines"]),
    }


def screen_document(doc_image, selfie_image):
    t_start = time.time()
    if doc_image is None:
        raise gr.Error("Upload a document image first.")

    doc_path = doc_image  # gradio gives temp path as str
    img = Image.open(doc_path).convert("RGB")

    forge = forge_detector.analyze_document(img)
    stamps = stamp_forensics.analyze_stamps(img)
    meta = metadata_analysis.analyze_metadata(doc_path)
    ocr = ocr_fields.extract_fields(img)

    face = None
    f = ocr["fields"]
    face_not_applicable = f.get("doc_type") == "pan"  # PAN cards carry no photograph
    if face_verify is not None and not face_not_applicable:
        face = face_verify(img, selfie_image)
        face_risk = face["risk"]
        if selfie_image is None:
            # No selfie to compare against: keep the morph/quality signal only
            face_risk = min(face["risk"], 0.35)
    elif face_not_applicable:
        face = {"document_face_detected": None,
                "findings": ["PAN is a signature-only document - face verification not applicable."]}
        face_risk = 0.0
    else:
        face_risk = 0.35
        face = {"document_face_detected": None, "findings": ["Face module unavailable"]}

    module_risks = {
        "forgery": forge["risk"],
        "stamp": stamps["risk"],
        "metadata": meta["risk"],
        "face": face_risk,
        "ocr": ocr["risk"],
    }

    # Module 2 - document validation + Module 4b - identity linkage
    val = document_validation.validate(
        name=f.get("name"), pan_number=f.get("pan_number"),
        aadhaar_number=f.get("aadhaar_number"), dob=f.get("dob"),
        expiry=f.get("expiry"), gender=f.get("gender"),
        doc_type=f.get("doc_type"), mrz_line1=f.get("mrz_line1"),
        mrz_line2=f.get("mrz_line2"),
    )
    module_risks["validation"] = 0.0
    if val["risk"]:
        module_risks["validation"] = val["risk"]

    linkage = {"matched": None, "multi_identity_flag": None, "note": "Selfie required for linkage check"}
    if face is not None and face.get("doc_embedding"):
        linkage = watchlist.check(face["doc_embedding"], f.get("name"), f.get("pan_number") or f.get("aadhaar_number"))
        if face_risk < 0.5:
            watchlist.add(face["doc_embedding"], f.get("name"), f.get("pan_number") or f.get("aadhaar_number"))

    fused = risk.fuse(module_risks, extra={"selfie_provided": bool(selfie_image)})
    trail_path = risk.write_trail(
        {"summary": _summary(ocr)},
        fused,
        doc_path=doc_path,
    )

    annot = _annotate(img, stamps["stamps"], None)

    # field findings
    module_lines = []
    module_lines.append(f"**Forgery ViT (ELA-blended):** forged prob = {forge['forge_probability']:.2f}  →  {forge['verdict']}")
    if forge["models"].get("ela_cnn_casia2") is not None:
        module_lines.append(f"**Dual RGB+ELA CNN (CASIA v2):** forged prob = {forge['models']['ela_cnn_casia2']:.2f}")
    module_lines.append(f"**Stamp/seal regions:** {stamps['stamp_count']} localized")
    for s in stamps["stamps"][:3]:
        module_lines.append(f"  - stamp @ {s['bbox']} circularity {s['circularity']:.2f}, ELA mean {s['ela_mean']:.1f}, susp {s['suspicious_score']:.2f}")
    module_lines.append(f"**Metadata:** {'EXIF present' if meta['has_exif'] else 'no EXIF'} | software={meta.get('software')}")
    module_lines.extend(f"  - {x}" for x in meta["findings"][:4])
    module_lines.append(f"**Face:** doc-detect={face['document_face_detected']}, sim={face.get('similarity')}, match={face.get('match')}")
    module_lines.extend(f"  - {x}" for x in face.get("findings", [])[:4])
    module_lines.append(f"**OCR:** DOB={ocr['fields']['dob']} PAN={ocr['fields']['pan_number']} Name={ocr['fields']['name']}")
    module_lines.extend(f"  - {x}" for x in ocr["findings"][:3])
    module_lines.append(f"**Validation ({val['doc_type']}):** expired={val['expired']} blacklist_hit={val['blacklist_hit']}")
    module_lines.extend(f"  - {x}" for x in val["rules_failed"][:3])
    if val["mrz"]:
        module_lines.append(f"  - MRZ valid={val['mrz']['mrz_valid']} | {val['mrz']['surname']}, {val['mrz']['given_names']}")
    module_lines.append(f"**Identity linkage:** {linkage['note']}")
    if linkage.get("matched"):
        module_lines.append(f"  - best prior match sim={linkage['matched']['similarity']:.3f} ({linkage['matched']['name']})")
    module_lines.append(f"**Trail:** {trail_path}")

    if face and face.get("doc_face_bbox"):
        annot = _draw_face(annot, face["doc_face_bbox"])

    return (
        fused["verdict"],
        round(fused["total_risk"], 4),
        {m: f"{d['risk']:.2f}" for m, d in fused["modules"].items()},
        _summary(ocr),
        "\n".join(module_lines),
        annot,
        round(time.time() - t_start, 1),
        trail_path,
    )


examples = []
_md = os.path.join(MORPH_DIR, "fake_0_0.jpg")
_le = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_doc1 = os.path.join(_le, "aadhar_labeled_imgs", "1.png") if os.path.isdir(_le) else None
if os.path.exists(_md):
    examples.append([_md, None])
if _doc1 and os.path.exists(_doc1):
    examples.append([_doc1, None])

with gr.Blocks(title="AI Fake Identity & Document Screening - SIH 2026") as demo:
    gr.Markdown(
        """
# AI-Based Fake Identity & Document Screening System
**SIH 2026 · Ministry of Home Affairs (SSB)** · *prétrained models only — no training*
`zodumair/document-forgery-detector` (ViT) · `PaddleOCR PP-OCRv6` · `InceptionResnetV1(VGGFace2)` · EXIF forensics
"""
    )
    with gr.Row():
        with gr.Column():
            doc_in = gr.Image(label="Document image (PAN / Aadhaar / ID)", type="filepath")
            selfie_in = gr.Image(label="Live selfie (optional - enables face verification)", type="filepath")
            run_btn = gr.Button("Run Screening", variant="primary")
        with gr.Column():
            verdict = gr.Textbox(label="Verdict", interactive=False)
            total = gr.Number(label="Total risk (0..1)", interactive=False)
            elapsed = gr.Number(label="Elapsed (s)", interactive=False)
            modules_out = gr.JSON(label="Per-module risk")
            trail_out = gr.Textbox(label="Digital trail (JSON path)")
    with gr.Row():
        with gr.Column():
            summary = gr.JSON(label="Extracted fields")
            module_notes = gr.Markdown("### Module details")
        with gr.Column():
            annot_img = gr.Image(label="Annotated document", interactive=False)

    gr.Examples(examples, inputs=[doc_in, selfie_in], cache_examples=False)
    run_btn.click(screen_document, inputs=[doc_in, selfie_in],
                  outputs=[verdict, total, modules_out, summary, module_notes, annot_img, elapsed, trail_out])

    gr.Markdown(f"**Model registry (all pretrained):** `{MODELS}`")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
"""Module - Document field extraction (PaddleOCR, pretrained PP-OCRv6).

Extracts and validates identity fields (PAN, DOB, Name, Aadhaar, signature)
from a scanned/printed identity document. Also returns raw OCR lines+boxes for
GT-localization comparisons (used by scripts/validate_fields.py).
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

from config import PASS_INDICATOR_TEXT

_PAN_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])")
_DOB_RE = re.compile(r"\b\d{1,2}\s?[/.-]\s?\d{1,2}\s?[/.-]\s?\d{2,4}\b")
_AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
_HINDI_DOB = re.compile(r"(?:जन्म\s*tिथि|DOB|Date\s*of\s*Birth)", re.I)

_TITLE_HINT = ("mrs", "mr.", "ms.", "smt", "shri", "kumari", "dr", "s/d", "d/o", "w/o", "h/o", "c/o",
               "d", "s", "k", "g", "v", "m", "r")
_BOILERPLATE = {
    "income", "tax", "department", "govt", "government", "of", "india", "permanent", "account",
    "number", "signature", "sign", "pan", "aadhaar", "aadhar", "address", "father", "mother",
    "republic", "republik", "welfare", "finance", "legislative", "female", "male", "gender",
    "isar", "asa", "en", "rgrsta", "a", "the", "and",
}


def _flat_lines(res) -> list[dict]:
    lines = []
    for r in res:
        texts = r.get("rec_texts") or []
        boxes = r.get("rec_polys") or r.get("rec_boxes") or []
        scores = r.get("rec_scores") or []
        for t, b, s in zip(texts, boxes, scores):
            t = re.sub(r"\s+", " ", str(t)).strip()
            if not t:
                continue
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            lines.append(
                {"text": t, "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))], "conf": float(s)}
            )
    return lines


def extract_fields(image) -> dict:
    """OCR an identity document and pull structured fields + full OCR lines."""
    from modules.models import get_ocr

    img = image if isinstance(image, Image.Image) else Image.open(image)
    path = None
    if not isinstance(image, Image.Image) and not hasattr(image, "save"):
        import os

        path = str(image)
    if path is None:
        res = get_ocr().predict(np.array(img.convert("RGB")))
    else:
        res = get_ocr().predict(path)
    lines = _flat_lines(res)

    joined = " ".join(l["text"] for l in lines)
    pan = _PAN_RE.search(joined.replace(" ", "").upper())
    if pan is None:
        # OCR-confusion tolerant attempt: O->0, I/L->1 (common on Aadhaar/PAN scans)
        for tbl in ({"O": "0"}, {"I": "1", "L": "1"}, {"O": "0", "I": "1", "L": "1"}):
            t = joined.replace(" ", "").upper()
            for a, b in tbl.items():
                t = t.replace(a, b)
            m = _PAN_RE.search(t)
            if m:
                pan = m
                break
    dob = _DOB_RE.search(joined)
    aadhaar = _AADHAAR_RE.search(joined.replace(" ", ""))
    has_signature = any(any(h in l["text"].lower() for h in PASS_INDICATOR_TEXT) for l in lines)

    # Name heuristics: first two non-boilerplate capitalized text lines,
    # preferring lines close to the DOB block and skipping official headers.
    def _nameworthy(l: dict) -> str | None:
        t = re.sub(r"[^A-Za-z .'-]", "", l["text"])
        if len(t) < 3 or re.search(r"[0-9]", l["text"]):
            return None
        words = [w for w in t.split() if w]
        if not words:
            return None
        alpha_words = [w.strip(".'-") for w in words if w.strip(".'-").isalpha()]
        if len(alpha_words) < 1:
            return None
        # reject OCR garble like "FarHTST" / "3TRTaR" (non-uniform case),
        # allow ALL-CAPS tokens and proper title-case tokens.
        for w in alpha_words:
            if w.isupper() or w.istitle():
                continue
            if len(w) == 1 and w in _TITLE_HINT:
                continue
            return None
        if alpha_words and all(w.lower() in _BOILERPLATE for w in alpha_words):
            return None
        return t

    dob_line_idx = None
    for i, l in enumerate(lines):
        if l["text"] and (re.search(r"जन्म", l["text"]) or re.search(r"\bDOB\b", l["text"].upper())):
            dob_line_idx = i
            break

    name_pool = []
    for i, l in enumerate(lines):
        t = _nameworthy(l)
        if t is None:
            continue
        name_pool.append((i, t))
    name_cand = []
    if dob_line_idx is not None:
        ordered = sorted(name_pool, key=lambda it: (abs(it[0] - dob_line_idx), it[0]))
    else:
        ordered = name_pool
    # skip lines far from DOB (top headers) unless nothing else exists
    for i, t in ordered:
        nwrds = [w.strip(".'-") for w in t.split() if w.strip(".'-").isalpha() and len(w.strip(".'-")) >= 4]
        if nwrds and t not in name_cand:
            name_cand.append(t)
        if len(name_cand) >= 2:
            break
    name = " ".join(name_cand[:2]) if name_cand else None

    # Full-line signature detection (handwriting-ish, low confidence OCR)
    sig_lines = []
    for l in lines:
        if l["conf"] < 0.55 and not any(re.search(r"[0-9]", ch) for ch in l["text"]) and len(l["text"]) >= 3:
            sig_lines.append(l["text"])
    signature_text = " ".join(sig_lines[:3]) or (name if name else None)

    # --- document-type detection + travel-doc fields (passport/visa) ---
    low = joined.lower()
    mrz_lines = []
    for l in lines:
        t = l["text"]
        if re.fullmatch(r"[A-Z0-9<]+", t) and len(t) >= 40:
            mrz_lines.append(t)
    mrz_line1 = mrz_lines[0] if len(mrz_lines) >= 1 else None
    mrz_line2 = mrz_lines[1] if len(mrz_lines) >= 2 else None

    doc_type = "unknown"
    if re.search(r"passport|<{40}", low) or mrz_line1:
        doc_type = "passport"
    elif "visa" in low or "permission to stay" in low:
        doc_type = "visa"
    elif "permanent account number" in low or _PAN_RE.search(joined.replace(" ", "").upper()):
        doc_type = "pan"
    elif "aadhaar" in low or "uidai" in low:
        doc_type = "aadhaar"
    elif "driving" in low or "licence" in low or "license" in low:
        doc_type = "driving_license"

    # expiry heuristic: a date appearing near an "expiry/expires" token
    expiry = None
    for i, l in enumerate(lines):
        if re.search(r"expir|valid until|valid thru", l["text"].lower()):
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(lines):
                    m = re.search(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", lines[j]["text"])
                    if m:
                        expiry = m.group(0)
                        break
            if expiry:
                break
    gender = None
    m = re.search(r"\b(MALE|FEMALE|M|F)\b", low.upper())
    if m and m.group(0) in ("MALE", "FEMALE"):
        gender = m.group(0).title()
    elif m and m.group(0) in ("M", "F") and re.search(r"sex|gender", low):
        gender = "Male" if m.group(0) == "M" else "Female"

    fields = {
        "pan_number": (pan.group(0) if pan else None),
        "dob": (dob.group(0) if dob else None),
        "aadhaar_number": (aadhaar.group(0) if aadhaar else None),
        "name": name,
        "has_signature": has_signature,
        "signature_text": signature_text,
        "doc_type": doc_type,
        "expiry": expiry,
        "gender": gender,
        "mrz_line1": mrz_line1,
        "mrz_line2": mrz_line2,
    }

    # Risk: missing critical fields on an identity doc is suspicious
    risk = 0.0
    findings = []
    if not fields["pan_number"] and not fields["aadhaar_number"]:
        findings.append("No PAN or Aadhaar number found on document")
        risk += 0.4
    if not fields["dob"]:
        findings.append("No date-of-birth found")
        risk += 0.25
    if not has_signature:
        findings.append("No signature/seal indicator found")
        risk += 0.15
    risk = min(1.0, risk)

    return {
        "fields": fields,
        "lines": lines,
        "risk": round(risk, 4),
        "findings": findings,
    }
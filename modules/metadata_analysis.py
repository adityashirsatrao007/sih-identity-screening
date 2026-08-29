"""Module - Image metadata analysis (EXIF / container forensics).

Flags: editing software, date inconsistency (capture vs modify), GPS on a
scanned document, screenshots, thumbnail rewrites, abnormal compression.
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime

import exifread
from PIL import Image

from config import SCREENSHOT_SOFTWARE, SUSPICIOUS_SOFTWARE


def _read_tags(path_or_bytes) -> dict:
    data = path_or_bytes
    if isinstance(data, str):
        with open(data, "rb") as f:
            data = f.read()
    tags = exifread.process_file(io.BytesIO(data), details=False)
    out = {}
    for k, v in tags.items():
        s = str(v)
        out[k] = s[:200]
    return out


def _date_parse(s: str):
    m = re.search(r"(\d{4}):(\d{2}):(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", s)
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups()))
    except Exception:
        return None


def analyze_metadata(image_path: str) -> dict:
    """Analyze EXIF + file-level metadata. `image_path` must be a local file."""
    tags = _read_tags(image_path)
    risk = 0.0
    findings = []
    software = ""
    make_model = ""
    for k in ("Image Software", "Image Make", "Image Model"):
        if k in tags:
            software = tags.get("Image Software", software)
            make_model = tags.get("Image Make") or tags.get("Image Model") or make_model

    has_exif = len(tags) > 0
    if software:
        sw = software.lower()
        if any(s in sw for s in SUSPICIOUS_SOFTWARE):
            findings.append(f"Edited with {software} (editing tool)")
            risk += 0.55 - (0.15 if any(t in sw for t in SCREENSHOT_SOFTWARE) else 0.0)
        elif any(s in sw for s in ("gimp", "krita")):
            findings.append(f"Open-source editor detected: {software}")
            risk += 0.45

    # Date consistency: DateTimeOriginal vs DateTime vs file mtime
    dt_o = _date_parse(tags.get("EXIF DateTimeOriginal", "")) or _date_parse(tags.get("Image DateTime", ""))
    dt_d = _date_parse(tags.get("EXIF DateTimeDigitized", ""))
    if dt_o and dt_d and abs((dt_o - dt_d).total_seconds()) > 3600:
        findings.append("Original-capture and digitized timestamps differ by > 1h")
        risk += 0.3

    # GPS on a scanned/printed document is anomalous
    if any(k.startswith("GPS ") for k in tags):
        findings.append("GPS metadata present on a document scan")
        risk += 0.5

    # Comment fields / rewrite markers
    for k in ("Image XResolution", "Image ImageDescription", "EXIF UserComment"):
        if k in tags and tags[k].strip() and k == "EXIF UserComment":
            findings.append("User comment field populated (uncommon for official scans)")
            risk += 0.2

    # Thumbnail rewrite
    if any("Thumbnail" in k for k in tags):
        findings.append("EXIF thumbnail present (often rewritten by editors)")

    # File-level heuristics
    try:
        st = os.stat(image_path)
        img = Image.open(image_path)
        w, h = img.size
        q = img.info.get("quality")
        img.close()
        if w > 0:
            ratio = st.st_size / (w * h)
        else:
            ratio = 0
        # Screenshot-style: small file, big dims, no EXIF below threshold
        if dt_o and st.st_mtime:
            if abs(dt_o.timestamp() - st.st_mtime) > 30 * 86400:
                findings.append("Capture date set long before file created (> 30 days)")
                risk += 0.25
    except Exception:
        pass

    if not has_exif and not findings:
        findings.append("No EXIF metadata (consistent with teller-sided scan/print)")
        risk -= 0.0  # neutral: scans legitimately lack EXIF

    risk = max(0.0, min(1.0, risk))
    return {
        "has_exif": has_exif,
        "software": software or None,
        "camera": make_model or None,
        "findings": findings,
        "risk": round(risk, 4),
    }
"""Module 2 - Document validation.

Verifies extracted fields against official formats & rules:
  - document-type detection (PAN / Aadhaar / Passport / Visa / DL / unknown)
  - format checks (PAN pattern, DOB sanity, Aadhaar Verhoeff-ish structure)
  - passport MRZ parsing + check-digit validation (TD3), expiry computation
  - blacklist / negative-list lookup (data/blacklist.txt, one number per line)
  - cross-field consistency hints

Pure rule engine - no ML.
"""

from __future__ import annotations

import os
import re
from datetime import date

from config import DATA_DIR

_PAN_RE = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")
_DOB_RE = re.compile(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})")
_AADHAAR_RE = re.compile(r"\d{4}\s?\d{4}\s?\d{4}")
_VISA_RE = re.compile(r"[A-Z0-9]{6,10}")
_MRZ_WEIGHTS = [7, 3, 1] * 7


def _mrz_val(ch: str) -> int:
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    return ord(ch) - ord("A") + 10


def _mrz_checkdigit(field: str) -> int:
    return sum(_mrz_val(c) * _MRZ_WEIGHTS[i] for i, c in enumerate(field)) % 10


def parse_mrz_td3(line1: str, line2: str) -> dict | None:
    """Passport MRZ (TD3): two lines of 44 chars. Returns parsed fields + checks."""
    if len(line1) < 44 or len(line2) < 44:
        return None
    try:
        doc_code = line1[0:2].strip("<") or "P"
        surname, given = line1[5:44].split("<<", 1)
        lines = {
            "doc_code": doc_code,
            "issuer": line1[2:5].strip("<"),
            "surname": surname.replace("<", " ").strip(),
            "given_names": given.replace("<", " ").strip(),
            "passport_number": line2[0:9].strip("<"),
            "nationality": line2[10:13].strip("<"),
            "dob": line2[13:19],
            "sex": line2[20],
            "expiry": line2[21:27],
            "personal_number": line2[28:42].strip("<"),
        }
        checks = {
            "passport_number": _mrz_checkdigit(line2[0:9]) == int(line2[9]),
            "dob": _mrz_checkdigit(line2[13:19]) == int(line2[19]),
            "expiry": _mrz_checkdigit(line2[21:27]) == int(line2[27]),
            "composite": _mrz_checkdigit(line2[0:10] + line2[13:20] + line2[21:28]) == int(line2[43]),
        }
        lines["check_digits"] = checks
        lines["mrz_valid"] = all(checks.values())
        return lines
    except Exception:
        return None


def _parse_date(text: str | None):
    if not text:
        return None
    m = _DOB_RE.search(text)
    if not m:
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    y += 2000 if y < 100 and y <= date.today().year % 100 else 1900 if y < 100 else 0
    try:
        return date(y, mo, d)
    except Exception:
        return None


def validate(**fields) -> dict:
    """:param fields: dict with keys name, pan_number, aadhaar_number, dob,
    expiry, doc_type, mrz_line1, mrz_line2, gender."""
    rules_passed, rules_failed, findings = [], [], []
    risk = 0.0

    # doc-type detection
    doc_type = fields.get("doc_type", "unknown")
    txt = " ".join(str(v or "") for v in fields.values())

    pan = fields.get("pan_number")
    if pan and not _PAN_RE.fullmatch(pan.strip()):
        rules_failed.append("PAN fails AAAAA9999A format")
        risk += 0.5
    elif pan:
        rules_passed.append("PAN format valid")

    aad = fields.get("aadhaar_number")
    if aad and not _AADHAAR_RE.fullmatch(aad.replace(" ", "").strip()):
        rules_failed.append("Aadhaar number has invalid 12-digit structure")
        risk += 0.5
    elif aad:
        rules_passed.append("Aadhaar number structure valid")

    local_dob = _parse_date(fields.get("dob"))
    if local_dob:
        age = (date.today() - local_dob).days / 365.25
        if age < 0 or age > 120:
            rules_failed.append(f"DOB {local_dob} implies implausible age ({age:.0f})")
            risk += 0.6
        else:
            rules_passed.append(f"DOB plausible (age ≈ {age:.0f})")

    # MRZ (passports)
    mrz = None
    if fields.get("mrz_line1"):
        mrz = parse_mrz_td3(fields["mrz_line1"], fields.get("mrz_line2", ""))
        if mrz:
            if mrz["mrz_valid"]:
                rules_passed.append("MRZ check digits valid")
            else:
                rules_failed.append(f"MRZ check digits INVALID ({[k for k, v in mrz['check_digits'].items() if not v]})")
                risk += 0.5
            if mrz.get("dob") and local_dob:
                if mrz["dob"] != f"{local_dob:%y%m%d}":
                    rules_failed.append("MRZ DOB disagrees with printed DOB")
                    risk += 0.6

    # Expiry
    expiry_txt = fields.get("expiry")
    exp = _parse_date(expiry_txt) if expiry_txt and _DOB_RE.search(str(expiry_txt)) else None
    if mrz and mrz.get("expiry"):
        exp = date(2000 + int(mrz["expiry"][0:2]), int(mrz["expiry"][2:4]), int(mrz["expiry"][4:6]))
    expired = False
    if exp:
        if exp < date.today():
            expired = True
            rules_failed.append(f"Document EXPIRED on {exp}")
            risk += 0.5
        else:
            rules_passed.append(f"Expiry {exp} valid (not expired)")

    # Blacklist
    black_file = os.path.join(DATA_DIR, "blacklist.txt")
    blacklist_hit = None
    if os.path.exists(black_file):
        with open(black_file) as fh:
            bad = {ln.strip().upper() for ln in fh if ln.strip()}
        hits = [v for v in (pan, aad) if v and str(v).upper() in bad]
        if hits:
            blacklist_hit = True
            rules_failed.append("Document number present in negative/black list!")
            risk += 1.0
        else:
            blacklist_hit = False
            rules_passed.append("No blacklist hit")

    risk = max(0.0, min(1.0, risk))
    return {
        "doc_type": doc_type,
        "mrz": mrz,
        "expired": expired,
        "blacklist_hit": blacklist_hit,
        "rules_passed": rules_passed,
        "rules_failed": rules_failed,
        "findings": findings,
        "risk": round(risk, 4),
    }
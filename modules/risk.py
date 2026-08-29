"""Risk fusion + standardized verdict + digital trail (single source of truth)."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone

from config import RISK_WEIGHTS, STRONG_SIGNAL_OVERRIDE, TRAIL_DIR, VERDICT


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def verdict_from_risk(total: float) -> str:
    if total < VERDICT["pass"]:
        return "PASS - genuine"
    if total <= VERDICT["review"]:
        return "SCREEN - manual review advised"
    return "REJECT - high risk of fake/fraudulent document"


def fuse(module_results: dict, extra: dict | None = None) -> dict:
    """Weighted fusion with a strong-signal override. Results are {module: risk}."""
    weighted = {}
    for key, w in RISK_WEIGHTS.items():
        r = float(module_results.get(key, 0.0))
        weighted[key] = {"risk": round(r, 4), "weight": w, "contribution": round(r * w, 4)}
    total = sum(v["contribution"] for v in weighted.values())

    overrides = []
    for key, th in STRONG_SIGNAL_OVERRIDE.items():
        if weighted[key]["risk"] >= th:
            total = max(total, VERDICT["review"] + 0.01)
            overrides.append(key)
    total = round(min(1.0, max(0.0, total)), 4)
    verdict = verdict_from_risk(total)
    return {
        "total_risk": total,
        "verdict": verdict,
        "modules": weighted,
        "strong_signal_overrides": overrides,
        "notes": extra or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_trail(case: dict, result: dict, doc_path: str | None = None, hash: str | None = None) -> str:
    """Persist a JSON audit trail for investigation / intelligence analysis."""
    cid = f"SIH-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    entry = {
        "case_id": cid,
        "created": datetime.now(timezone.utc).isoformat(),
        "document_sha256": hash or (_sha(doc_path) if doc_path else None),
        "input_summary": case.get("summary", {}),
        "result": result,
    }
    fp = os.path.join(TRAIL_DIR, f"{cid}.json")
    with open(fp, "w") as f:
        json.dump(entry, f, indent=2, default=str)
    return fp
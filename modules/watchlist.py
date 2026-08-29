"""Multi-identity / repeat-traveller watchlist (part of Module 4 + digital trail).

Keeps the last N screened face embeddings + identities locally and flags:
  - same face, different name   -> possible MULTI-IDENTITY (identity switching)
  - same face, same name        -> repeat traveller (benign, historical context)
Thresholds tuned for facenet (VGGFace2) cosine similarity space.
"""

from __future__ import annotations

import json
import os

import numpy as np

from config import DATA_DIR

WATCH_FILE = os.path.join(DATA_DIR, "watchlist.json")
MAX_ENTRIES = 500
SIM_THRESHOLD = 0.5  # same-face threshold (facenet space)


def _load() -> list:
    if os.path.exists(WATCH_FILE):
        try:
            with open(WATCH_FILE) as fh:
                return json.load(fh)
        except Exception:
            return []
    return []


def _save(entries: list):
    with open(WATCH_FILE, "w") as fh:
        json.dump(entries[-MAX_ENTRIES:], fh, indent=None)


def check(embedding: list | np.ndarray, name: str, doc_number: str | None = None) -> dict:
    """Compare a fresh face embedding against prior screenings."""
    emb = np.asarray(embedding, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    entries = _load()
    best = None
    for e in entries:
        try:
            e_emb = np.asarray(e["embedding"], dtype=np.float32)
            e_emb = e_emb / (np.linalg.norm(e_emb) + 1e-9)
            sim = float(np.dot(emb, e_emb))
            if best is None or sim > best["similarity"]:
                best = {"name": e.get("name"), "doc_number": e.get("doc_number"),
                        "when": e.get("when"), "similarity": sim}
        except Exception:
            continue

    multi_identity = None
    if best and best["similarity"] >= SIM_THRESHOLD:
        same_name = (name or "").strip().upper() == (best.get("name") or "").strip().upper()
        multi_identity = not same_name
    return {
        "matched": best,
        "multi_identity_flag": multi_identity,
        "note": ("IDENTITY MISMATCH DETECTED - same face as prior screening under a different name!" if multi_identity is True
                 else "Repeat traveller (same identity previously screened)" if best and best["similarity"] >= SIM_THRESHOLD
                 else "No prior identity linkage"),
    }


def add(embedding: list | np.ndarray, name: str, doc_number: str | None = None, family: str = ""):
    """Register a screened embedding for future linkage checks."""
    if embedding is None:
        return
    emb = np.asarray(embedding, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    entries = _load()
    entries.append({
        "name": name or "UNKNOWN",
        "doc_number": doc_number,
        "embedding": emb.tolist(),
        "when": __import__("datetime").datetime.now().isoformat(),
    })
    _save(entries)
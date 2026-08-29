"""Batch validation: run the forgery detector over all 500 dataset morph images.

Every Mendeley sample (k7srr5wwhv) is a faceswapper.AI-morphed Aadhaar image,
so a correct detector should flag the vast majority as forged. Outputs
detection rate, score distribution and per-image timing.

Run: python scripts/validate_morph.py
"""

from __future__ import annotations

import glob
import os
import time
import warnings
from statistics import mean, stdev

warnings.filterwarnings("ignore")
os.environ.setdefault("FORGE_KERAS", "1")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

from config import MORPH_DIR
from modules import forge_detector
from modules.models import ela_residual_stats
from PIL import Image


def main(sample: int | None = None):
    files = sorted(glob.glob(os.path.join(MORPH_DIR, "*.jpg")))
    if sample:
        files = files[:sample]
    print(f"Validating {len(files)} morphed images from {MORPH_DIR}")

    t0 = time.time()
    probs, risks, elas = [], [], {}
    forged_hits = 0
    for i, f in enumerate(files, 1):
        img = Image.open(f).convert("RGB")
        res = forge_detector.analyze_document(img)
        st = ela_residual_stats(img)
        p = res["forge_probability"]
        probs.append(p)
        risks.append(res["risk"])
        for k in ("ela_mean", "ela_std"):
            elas.setdefault(k, []).append(st[k])
        if p >= 0.5:
            forged_hits += 1
        if i % 50 == 0:
            print(f"  {i}/{len(files)}  forged_rate_so_far={forged_hits / i * 100:.1f}%")
    dt = time.time() - t0

    print("\n===== RESULTS =====")
    print(f"Images          : {len(files)}")
    print(f"Detected forged : {forged_hits}  ({forged_hits / len(files) * 100:.1f}%)")
    print(f"P(forged) mean  : {mean(probs):.4f}  (stdev {stdev(probs):.4f})")
    print(f"P(forged) min   : {min(probs):.4f}   max {max(probs):.4f}")
    print(f"P(forged) >=0.9 : {sum(1 for p in probs if p >= 0.9) / len(probs) * 100:.1f}%")
    print(f"ELA mean        : {mean(elas['ela_mean']):.1f}  (stdev {stdev(elas['ela_mean']):.1f})")
    print(f"Total time      : {dt:.1f}s  ({dt / max(len(files), 1):.3f}s/image)")
    print("NOTE: dataset contains ONLY forged (morphed) images; high detection rate = good.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="run on first N images only")
    args = ap.parse_args()
    main(args.sample)
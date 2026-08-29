"""Batch validation: OCR field extraction vs Label-Studio ground-truth boxes.

The dataset (aadhar_labeled) is 31 identity documents with YOLO-format boxes
for classes {0:name, 1:dob, 2:pan_number, 3:signature}. We run the pretrained
PaddleOCR pipeline and measure (a) how well OCR text boxes overlap GT field
boxes (mean IoU / coverage) and (b) regex extraction rate for PAN/DOB.

Run: python scripts/validate_fields.py [--gt labels-2|labels-high-res]
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from statistics import mean

warnings.filterwarnings("ignore")
os.environ.setdefault("FLAGS_enable_pir_api", "0")

from config import LABELED_GT_DIR, LABELED_GT_HR_DIR, LABELED_IMG_DIR
from modules import ocr_fields
from PIL import Image

CLASSES = {0: "name", 1: "dob", 2: "pan_number", 3: "signature"}


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    i = ix * iy
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - i
    return i / u if u > 0 else 0.0


def _cov(a, b):
    """fraction of GT box b covered by OCR box a"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    garea = (bx2 - bx1) * (by2 - by1)
    return (ix * iy) / garea if garea > 0 else 0.0


def main(gt_name: str = "labels-2"):
    gt_dir = LABELED_GT_DIR if gt_name == "labels-2" else LABELED_GT_HR_DIR
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    gt_files = [f for f in gt_files if not os.path.basename(f) == "classes.txt"]
    print(f"GT dir: {gt_name} with {len(gt_files)} files")

    per_class_iou = {c: [] for c in CLASSES.values()}
    per_class_cov = {c: [] for c in CLASSES.values()}
    pan_hits = dob_hits = sig_found = total_pan = total_dob = 0

    for gf in gt_files:
        stem = os.path.splitext(os.path.basename(gf))[0]
        img_path = os.path.join(LABELED_IMG_DIR, f"{stem}.png")
        if not os.path.exists(img_path):
            continue
        W, H = Image.open(img_path).size

        boxes_gt = []
        with open(gf) as fh:
            for line in fh:
                line = line.strip()
                if not line or not line[0].isdigit():
                    continue
                parts = line.split()
                c, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
                boxes_gt.append((CLASSES.get(c, f"class{c}"), (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))

        res = ocr_fields.extract_fields(img_path)
        ocr_boxes = [l["box"] for l in res["lines"]]

        for cls, (gx, gy, gw, gh) in boxes_gt:
            gbox = (gx * W, gy * H, (gx + gw) * W, (gy + gh) * H)
            if cls not in per_class_iou:
                continue
            best_iou, best_cov = 0.0, 0.0
            for ob in ocr_boxes:
                iou = _iou(ob, gbox)
                cov = _cov(ob, gbox)
                best_iou = max(best_iou, iou)
                best_cov = max(best_cov, cov)
            per_class_iou[cls].append(best_iou)
            per_class_cov[cls].append(best_cov)

        f = res["fields"]
        if f["pan_number"]:
            pan_hits += 1
        if f.get("dob"):
            dob_hits += 1
        if f.get("has_signature"):
            sig_found += 1
        if any(c == "pan_number" for c, _ in boxes_gt):
            total_pan += 1
        if any(c == "dob" for c, _ in boxes_gt):
            total_dob += 1

    print("\n===== GT-box localization (OCR text box vs labelled field box) =====")
    for cls in CLASSES.values():
        if per_class_iou[cls]:
            print(f"  {cls:12s} mean_IoU={mean(per_class_iou[cls]):.3f}  mean_coverage={mean(per_class_cov[cls]):.3f}")
    print("\n===== Regex / semantic extraction =====")
    print(f"  PAN number detected : {pan_hits}/{total_pan}  ({pan_hits / total_pan * 100:.0f}%)")
    print(f"  DOB detected        : {dob_hits}/{total_dob}  ({dob_hits / total_dob * 100:.0f}%)")
    print(f"  Signature hint      : {sig_found}/{len(gt_files)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="labels-2", choices=["labels-2", "labels-high-res"])
    args = ap.parse_args()
    main(args.gt)
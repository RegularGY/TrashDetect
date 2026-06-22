# =============================================================
#  EcoSort AI — YOLO Retraining Script
#  retrain_yolo.py
#
#  Combines the original Dataset/ with YOLO_Feedback/
#  (auto-generated labels from user corrections) and
#  retrains the YOLOv8n detection model.
#
#  Run manually when enough feedback corrections are collected
#  (recommended minimum: 20 corrections with YOLO labels saved).
#
#  Usage:
#    python StaticDetectCode/retrain_yolo.py
#
#  Output:
#    runs/detect/ecosort_v4/weights/best.pt (or v5, v6...)
#
#  After retraining:
#    Update YOLO_PATH in StaticDetectCode/app.py to point
#    to the new best.pt, then restart Flask.
#
#  NOTE: YOLO retraining improves detection on BOTH static
#  image detection AND the live camera stream (since both
#  use YOLO for object localisation). CNN retraining only
#  improves static image detection and saved live camera
#  snapshots — not the continuous live stream.
#
#  LIMITATION: This retraining script only benefits from
#  corrections where YOLO correctly localised the bounding
#  box but assigned the wrong class. Cases where YOLO missed
#  the object entirely, detected ghost boxes, or produced
#  duplicate boxes cannot be addressed through this feedback
#  mechanism and require dedicated annotation-based retraining.
# =============================================================

import os
import sys
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent

# Original dataset (Roboflow export)
ORIGINAL_DATASET = BASE_DIR / "Dataset"

# YOLO feedback images and labels from user corrections
YOLO_FEEDBACK = BASE_DIR / "YOLO_Feedback"

# Combined dataset for retraining (temp folder)
COMBINED_DATASET = BASE_DIR / "Dataset_Retrain"

# Find latest YOLO model version to fine-tune from
RUNS_DIR = BASE_DIR / "runs" / "detect"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

existing_runs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith('ecosort_v')])
if existing_runs:
    last_ver = int(existing_runs[-1].name.split('_v')[-1])
    new_ver  = last_ver + 1
    prev_weights = existing_runs[-1] / "weights" / "best.pt"
else:
    new_ver      = 4
    prev_weights = BASE_DIR / "runs" / "detect" / "ecosort_v3" / "weights" / "best.pt"

RUN_NAME = f"ecosort_v{new_ver}"

# Training config
EPOCHS   = 50
IMG_SIZE = 416
BATCH    = 2
WORKERS  = 2
CLASSES  = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def retrain_yolo():

    print(f"\n{'=' * 60}")
    print(f"  EcoSort AI — YOLO Retraining ({RUN_NAME})")
    print(f"{'=' * 60}")

    # ── Validate ─────────────────────────────────────────────────
    if not ORIGINAL_DATASET.exists():
        print(f"\n[ERROR] Original Dataset not found at: {ORIGINAL_DATASET}")
        sys.exit(1)

    if not YOLO_FEEDBACK.exists() or not any(YOLO_FEEDBACK.glob("images/*.jpg")):
        print(f"\n[ERROR] No YOLO feedback images found at: {YOLO_FEEDBACK}")
        print(f"        Collect feedback corrections first via the web UI.")
        sys.exit(1)

    if not prev_weights.exists():
        print(f"\n[ERROR] Previous weights not found: {prev_weights}")
        sys.exit(1)

    # Count feedback samples
    fb_images = list((YOLO_FEEDBACK / 'images').glob("*.jpg"))
    fb_labels = list((YOLO_FEEDBACK / 'labels').glob("*.txt"))
    print(f"\n[INFO] Feedback images : {len(fb_images)}")
    print(f"[INFO] Feedback labels : {len(fb_labels)}")

    if len(fb_images) < 5:
        print(f"\n[WARNING] Only {len(fb_images)} feedback images found.")
        print(f"          Recommend collecting at least 20 before retraining.")
        confirm = input("Continue anyway? (y/N): ")
        if confirm.lower() != 'y':
            print("[INFO] Retraining cancelled.")
            sys.exit(0)

    # ── Build combined dataset ────────────────────────────────────
    print(f"\n[INFO] Building combined dataset at: {COMBINED_DATASET}")

    for split in ['train', 'valid']:
        (COMBINED_DATASET / split / 'images').mkdir(parents=True, exist_ok=True)
        (COMBINED_DATASET / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Copy original dataset split
        src_img = ORIGINAL_DATASET / split / 'images'
        src_lbl = ORIGINAL_DATASET / split / 'labels'
        if src_img.exists():
            for f in src_img.glob("*"):
                shutil.copy2(f, COMBINED_DATASET / split / 'images' / f.name)
        if src_lbl.exists():
            for f in src_lbl.glob("*"):
                shutil.copy2(f, COMBINED_DATASET / split / 'labels' / f.name)

    # Add feedback images/labels to training split only
    for img in fb_images:
        shutil.copy2(img, COMBINED_DATASET / 'train' / 'images' / img.name)
    for lbl in fb_labels:
        shutil.copy2(lbl, COMBINED_DATASET / 'train' / 'labels' / lbl.name)

    # Count combined totals
    train_total = len(list((COMBINED_DATASET / 'train' / 'images').glob("*")))
    valid_total = len(list((COMBINED_DATASET / 'valid' / 'images').glob("*")))
    print(f"[INFO] Combined train  : {train_total} images")
    print(f"[INFO] Combined valid  : {valid_total} images")

    # ── Write data.yaml ───────────────────────────────────────────
    data_yaml = {
        'path':  str(COMBINED_DATASET),
        'train': 'train/images',
        'val':   'valid/images',
        'nc':    len(CLASSES),
        'names': CLASSES,
    }
    yaml_path = COMBINED_DATASET / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"[INFO] data.yaml written: {yaml_path}")

    # ── Retrain YOLOv8 ───────────────────────────────────────────
    print(f"\n[INFO] Fine-tuning from: {prev_weights}")
    print(f"[INFO] Output run name : {RUN_NAME}")
    print(f"[INFO] Epochs          : {EPOCHS}")
    print(f"[INFO] Image size      : {IMG_SIZE}")
    print(f"[INFO] Batch size      : {BATCH}\n")

    model = YOLO(str(prev_weights))
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        name=RUN_NAME,
        project=str(RUNS_DIR),
        iou=0.45,
        exist_ok=True,
    )

    # ── Cleanup temp combined dataset ─────────────────────────────
    shutil.rmtree(COMBINED_DATASET, ignore_errors=True)
    print(f"\n[INFO] Temp dataset cleaned up.")

    new_weights = RUNS_DIR / RUN_NAME / "weights" / "best.pt"

    print(f"\n{'=' * 60}")
    print(f"  YOLO RETRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Version              : {RUN_NAME}")
    print(f"  Feedback images used : {len(fb_images)}")
    print(f"  New weights saved to : {new_weights}")
    print(f"{'=' * 60}")
    print(f"\n[NEXT STEP] Update YOLO_PATH in app.py:")
    print(f"  YOLO_PATH = r\"{new_weights}\"")
    print(f"  Then restart Flask.")
    print(f"\n[NOTE] YOLO retraining improves BOTH static image detection")
    print(f"       AND the live camera stream, since both use YOLO for")
    print(f"       object localisation.\n")


if __name__ == "__main__":
    retrain_yolo()

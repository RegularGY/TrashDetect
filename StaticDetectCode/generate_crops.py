# =============================================================
#  EcoSort AI — CNN Crop Generator
#  generate_crops.py
#
#  This script runs YOLO on your Phase 1, 2.5 and 3 training
#  images and saves cropped detections sorted by class.
#  These crops become the training data for ResNet18 CNN.
#
#  Usage:
#    python generate_crops.py
# =============================================================

import os
import sys
import cv2
import uuid
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

# Path to your trained YOLO model
MODEL_PATH = r"C:\Users\Victus\Documents\GitHub\TrashDetect\runs\detect\ecosort_v3\weights\best.pt"

# Source folders — Phase 1, 2.5 and 3 images only
# These are single-object phases so crops will be clean
SOURCE_FOLDERS = [
    r"C:\Users\Victus\Documents\GitHub\TrashDetect\Dataset\phase_raw\phase1_backup",
]

# If you don't have a phase1_backup folder, use the train/valid/test images directly
# The script will use the existing Dataset/train, valid, test folders
USE_SPLIT_DATASET = True  # Set True to use train/valid/test folders instead

SPLIT_FOLDERS = [
    r"C:\Users\Victus\Documents\GitHub\TrashDetect\Dataset\train\images",
    r"C:\Users\Victus\Documents\GitHub\TrashDetect\Dataset\valid\images",
    r"C:\Users\Victus\Documents\GitHub\TrashDetect\Dataset\test\images",
]

# Output folder for crops — sorted by class
CROPS_OUTPUT = r"C:\Users\Victus\Documents\GitHub\TrashDetect\CNN_Dataset"

# Minimum confidence for a detection to be saved as a crop
CONFIDENCE_THRESHOLD = 0.60

# Minimum crop size in pixels (width and height)
# Crops smaller than this are too small for CNN training
MIN_CROP_SIZE = 50

# Padding around bounding box (pixels)
# Adds a small border around the crop
PADDING = 10

# CNN input size
CNN_SIZE = 224

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def generate_crops():

    # Load YOLO model
    print(f"\n[INFO] Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Classes: {model.names}\n")

    # Create output class folders
    classes = list(model.names.values())
    for cls in classes:
        Path(CROPS_OUTPUT, cls).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created crop folders for: {classes}")
    print(f"[INFO] Output: {CROPS_OUTPUT}\n")

    # Get image paths
    if USE_SPLIT_DATASET:
        source_dirs = SPLIT_FOLDERS
    else:
        source_dirs = SOURCE_FOLDERS

    image_paths = []
    for folder in source_dirs:
        folder = Path(folder)
        if not folder.exists():
            print(f"[WARNING] Folder not found: {folder} — skipping.")
            continue
        paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        image_paths.extend(paths)
        print(f"[INFO] Found {len(paths)} images in {folder}")

    if not image_paths:
        print("[ERROR] No images found. Check your SOURCE_FOLDERS or SPLIT_FOLDERS paths.")
        sys.exit(1)

    print(f"\n[INFO] Total images to process: {len(image_paths)}")
    print(f"[INFO] Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"[INFO] Minimum crop size: {MIN_CROP_SIZE}px\n")

    # Counters
    total_crops  = 0
    class_counts = {cls: 0 for cls in classes}
    skipped      = 0

    # Process each image
    for idx, image_path in enumerate(image_paths, 1):

        if idx % 100 == 0 or idx == 1:
            print(f"[{idx}/{len(image_paths)}] Processing...")

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h, w = image.shape[:2]

        # Run YOLO detection
        results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                class_id   = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                # Get bounding box with padding
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(w, x2 + PADDING)
                y2 = min(h, y2 + PADDING)

                crop_w = x2 - x1
                crop_h = y2 - y1

                # Skip crops that are too small
                if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
                    skipped += 1
                    continue

                # Crop the image
                crop = image[y1:y2, x1:x2]

                # Resize to CNN input size
                crop_resized = cv2.resize(crop, (CNN_SIZE, CNN_SIZE))

                # Save crop to class folder
                crop_filename = f"{class_name}_{uuid.uuid4().hex[:8]}.jpg"
                crop_path     = Path(CROPS_OUTPUT) / class_name / crop_filename
                cv2.imwrite(str(crop_path), crop_resized)

                total_crops             += 1
                class_counts[class_name] += 1

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  CROP GENERATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Total crops saved : {total_crops}")
    print(f"  Skipped (too small): {skipped}")
    print(f"\n  Crops per class:")
    for cls, count in class_counts.items():
        print(f"    {cls:<12} : {count}")
    print(f"{'=' * 50}")
    print(f"\n[INFO] Crops saved to: {CROPS_OUTPUT}")
    print(f"[INFO] Next step: run train_cnn.py to train ResNet18\n")


if __name__ == "__main__":
    generate_crops()

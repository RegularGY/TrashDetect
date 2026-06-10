# =============================================================
#  EcoSort AI — Phase 1 Inference Script
#  detect.py
#
#  Usage:
#    python detect.py --source "path/to/image.jpg"
#    python detect.py --source "path/to/folder/"
#    python detect.py  (uses default Test_Image(Input)/ folder)
# =============================================================

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG — update these paths if needed
# ─────────────────────────────────────────────

# Path to your trained model weights
MODEL_PATH = r"C:\Users\Victus\Documents\GitHub\TrashDetect\runs\detect\ecosort_v3\weights\best.pt"

# Default input folderpython detect.py
INPUT_FOLDER = r"C:\Users\Victus\Documents\GitHub\TrashDetect\Test_Image(Input)"

# Output folder where results are saved
OUTPUT_FOLDER = r"C:\Users\Victus\Documents\GitHub\TrashDetect\Result(Output)"

# Minimum confidence threshold (0.0 - 1.0)
# Detections below this confidence will be ignored
CONFIDENCE_THRESHOLD = 0.50

# Bounding box and label colours per class
CLASS_COLOURS = {
    "cardboard": (255, 165,   0),   # Orange
    "glass":     ( 64, 224, 208),   # Turquoise
    "metal":     (192, 192, 192),   # Silver
    "paper":     (255, 255, 102),   # Yellow
    "plastic":   ( 51, 153, 255),   # Blue
}

DEFAULT_COLOUR = (0, 255, 0)  # Green fallback

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def draw_detection(image, box, class_name, confidence, colour):
    """
    Draws a bounding box and label on the image.
    Returns the annotated image.
    """
    x1, y1, x2, y2 = map(int, box)

    # Draw bounding box rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)

    # Build label text: "plastic 0.93"
    label = f"{class_name} {confidence:.2f}"

    # Calculate label background size
    font            = cv2.FONT_HERSHEY_SIMPLEX
    font_scale      = 1.2
    font_thickness  = 3
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Draw filled rectangle behind label for readability
    label_y = max(y1 - 10, text_h + 10)
    cv2.rectangle(
        image,
        (x1, label_y - text_h - baseline - 4),
        (x1 + text_w + 4, label_y + baseline - 4),
        colour,
        thickness=cv2.FILLED
    )

    # Draw label text in black on top of coloured background
    cv2.putText(
        image, label,
        (x1 + 2, label_y - 4),
        font, font_scale,
        (0, 0, 0),
        font_thickness,
        cv2.LINE_AA
    )

    return image


def print_summary(results_log):
    """Prints a summary table of all detections."""
    print("\n" + "=" * 55)
    print("  DETECTION SUMMARY")
    print("=" * 55)
    print(f"  {'Image':<25} {'Class':<12} {'Confidence':>10}")
    print("-" * 55)

    if not results_log:
        print("  No detections found.")
    else:
        for entry in results_log:
            print(f"  {entry['image']:<25} {entry['class']:<12} {entry['confidence']:>9.1%}")

    print("=" * 55)
    print(f"  Total detections: {len(results_log)}")
    print("=" * 55 + "\n")


def get_image_paths(source):
    """
    Returns a list of image file paths from a file or folder.
    """
    source = Path(source)

    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            return [source]
        else:
            print(f"[ERROR] File '{source}' is not a supported image format.")
            sys.exit(1)

    elif source.is_dir():
        paths = [
            p for p in source.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not paths:
            print(f"[ERROR] No images found in '{source}'")
            sys.exit(1)
        return sorted(paths)

    else:
        print(f"[ERROR] Source '{source}' does not exist.")
        sys.exit(1)


# ─────────────────────────────────────────────
#  MAIN DETECTION FUNCTION
# ─────────────────────────────────────────────

def run_detection(source, model_path, output_folder, conf_threshold):
    """
    Main detection pipeline:
    1. Load model
    2. Loop through images
    3. Run YOLO detection
    4. Draw bounding boxes with OpenCV
    5. Save to output folder
    6. Print results
    """

    # ── Load model ──────────────────────────
    print(f"\n[INFO] Loading model from: {model_path}")
    if not Path(model_path).exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print("        Check that MODEL_PATH in detect.py points to your best.pt file.")
        sys.exit(1)

    model = YOLO(model_path)
    print(f"[INFO] Model loaded successfully.")
    print(f"[INFO] Classes: {model.names}\n")

    # ── Prepare output folder ────────────────
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # ── Get image paths ──────────────────────
    image_paths = get_image_paths(source)
    print(f"[INFO] Found {len(image_paths)} image(s) to process.")
    print(f"[INFO] Confidence threshold: {conf_threshold:.0%}\n")

    results_log = []
    start_time  = time.time()

    # ── Process each image ───────────────────
    for idx, image_path in enumerate(image_paths, 1):

        print(f"[{idx}/{len(image_paths)}] Processing: {image_path.name}")

        # Read image with OpenCV
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  [WARNING] Could not read image: {image_path.name} — skipping.")
            continue

        # Run YOLO detection
        results = model(image, conf=conf_threshold, verbose=False)

        detection_count = 0

        # ── Loop through detections ──────────
        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                print(f"  [INFO] No detections in {image_path.name}")
                continue

            for box in boxes:
                # Get class ID and name
                class_id   = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                # Get bounding box coordinates (x1, y1, x2, y2)
                coords = box.xyxy[0].tolist()

                # Get colour for this class
                colour = CLASS_COLOURS.get(class_name, DEFAULT_COLOUR)

                # Draw on image
                image = draw_detection(image, coords, class_name, confidence, colour)

                # Log result
                results_log.append({
                    "image":      image_path.name,
                    "class":      class_name,
                    "confidence": confidence
                })

                detection_count += 1
                print(f"  ✓ Detected: {class_name} ({confidence:.1%} confidence)")

        # ── Save annotated image ─────────────
        output_filename = f"result_{image_path.name}"
        output_path     = Path(output_folder) / output_filename
        cv2.imwrite(str(output_path), image)
        print(f"  Saved → {output_path}\n")

    # ── Final summary ────────────────────────
    elapsed = time.time() - start_time
    print_summary(results_log)
    print(f"[INFO] Processing complete in {elapsed:.1f} seconds.")
    print(f"[INFO] Results saved to: {output_folder}\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="EcoSort AI — Phase 1 Trash Detection Inference Script"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=INPUT_FOLDER,
        help="Path to image file or folder (default: Test_Image(Input)/)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to trained model weights (default: ecosort_v2/weights/best.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FOLDER,
        help="Path to output folder (default: Result(Output)/)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold 0.0-1.0 (default: 0.50)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_detection(
        source        = args.source,
        model_path    = args.model,
        output_folder = args.output,
        conf_threshold= args.conf
    )

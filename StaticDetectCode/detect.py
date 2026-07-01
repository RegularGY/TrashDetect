# =============================================================
#  EcoSort AI — Hybrid YOLO + CNN Inference Script
#  detect.py
#
#  Phase 6: YOLO detects and localises objects,
#           ResNet18 CNN refines the classification.
#
#  Usage:
#    python detect.py --source "path/to/image.jpg"
#    python detect.py --source "path/to/folder/"
#    python detect.py  (uses default Test_Image(Input)/ folder)
#    ecosort-env\Scripts\activate
#    cd C:\Users\Victus\Documents\GitHub\TrashDetect (root for retrain)
#    python StaticDetectCode/retrain_cnn.py (retrain CNN)
#    python StaticDetectCode/retrain_yolo.py (retrain YOLO)
# =============================================================

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

# YOLO model path
MODEL_PATH = r"C:\Users\Victus\Documents\GitHub\TrashDetect\runs\detect\ecosort_v3\weights\best.pt"

# CNN model path
CNN_PATH = r"C:\Users\Victus\Documents\GitHub\TrashDetect\runs\cnn\ecosort_cnn_v1.pt"

# Default input folder
INPUT_FOLDER = r"C:\Users\Victus\Documents\GitHub\TrashDetect\Test_Image(Input)"

# Output folder
OUTPUT_FOLDER = r"C:\Users\Victus\Documents\GitHub\TrashDetect\Result(Output)"

# YOLO minimum confidence threshold
YOLO_CONFIDENCE = 0.50

# CNN confidence threshold
# If CNN confidence >= this value, CNN label overrides YOLO
# If CNN confidence < this value, YOLO label is kept
CNN_CONFIDENCE_THRESHOLD = 0.70

# CNN input size
CNN_SIZE = 224

# Preprocessing flags
ENABLE_NOISE_REMOVAL    = True
ENABLE_CONTRAST_ENHANCE = True
ENABLE_SHARPEN          = False

# Bounding box colours per class (BGR)
CLASS_COLOURS = {
    "cardboard": (0,   165, 255),   # Orange
    "glass":     (208, 224,  64),   # Turquoise
    "metal":     (192, 192, 192),   # Silver
    "paper":     (102, 255, 255),   # Yellow
    "plastic":   (255, 153,  51),   # Blue
}
DEFAULT_COLOUR = (0, 255, 0)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────

def load_yolo(model_path):
    """Load YOLOv8 detection model."""
    print(f"[INFO] Loading YOLO model from: {model_path}")
    if not Path(model_path).exists():
        print(f"[ERROR] YOLO model not found: {model_path}")
        sys.exit(1)
    model = YOLO(model_path)
    print(f"[INFO] YOLO loaded. Classes: {model.names}")
    return model


def load_cnn(cnn_path):
    """Load ResNet18 CNN classification model."""
    print(f"[INFO] Loading CNN model from: {cnn_path}")
    if not Path(cnn_path).exists():
        print(f"[WARNING] CNN model not found: {cnn_path}")
        print(f"[WARNING] Running in YOLO-only mode.")
        return None, None

    checkpoint = torch.load(cnn_path, map_location="cpu")
    classes    = checkpoint["classes"]
    num_classes = checkpoint["num_classes"]

    # Rebuild ResNet18 architecture
    model     = models.resnet18(weights=None)
    model.fc  = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    print(f"[INFO] CNN loaded. Classes: {classes}")
    print(f"[INFO] CNN best val accuracy: {checkpoint.get('best_val_acc', 'N/A'):.1%}")
    return model, classes


# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_image(image):
    """Apply OpenCV preprocessing before YOLO detection."""
    preprocessed = image.copy()
    if ENABLE_NOISE_REMOVAL:
        preprocessed = cv2.GaussianBlur(preprocessed, (3, 3), 0)
    if ENABLE_CONTRAST_ENHANCE:
        lab     = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        lab     = cv2.merge((l, a, b))
        preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if ENABLE_SHARPEN:
        blurred      = cv2.GaussianBlur(preprocessed, (0, 0), 3)
        preprocessed = cv2.addWeighted(preprocessed, 1.5, blurred, -0.5, 0)
    return preprocessed


# CNN image transform
cnn_transform = transforms.Compose([
    transforms.Resize((CNN_SIZE, CNN_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def classify_crop(crop_bgr, cnn_model, cnn_classes):
    """
    Run ResNet18 CNN on a cropped object image.
    Returns (class_name, confidence).
    """
    device = next(cnn_model.parameters()).device

    # Convert BGR crop to PIL RGB
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(crop_rgb)

    # Transform and run
    tensor = cnn_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs     = cnn_model(tensor)
        probs       = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_name = cnn_classes[predicted.item()]
    conf_value = confidence.item()

    return class_name, conf_value


# ─────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────

def draw_detection(image, box, class_name, yolo_conf, cnn_conf, model_used, colour):
    """
    Draws bounding box and label on the image.
    Shows which model made the final decision.
    """
    x1, y1, x2, y2 = map(int, box)

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)

    # Build label
    if model_used == "CNN":
        label = f"{class_name} {cnn_conf:.2f} [CNN]"
    else:
        label = f"{class_name} {yolo_conf:.2f} [YOLO]"

    font           = cv2.FONT_HERSHEY_SIMPLEX
    font_scale     = 1.2
    font_thickness = 3
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    label_y = max(y1 - 10, text_h + 10)
    cv2.rectangle(
        image,
        (x1, label_y - text_h - baseline - 4),
        (x1 + text_w + 4, label_y + baseline - 4),
        colour,
        thickness=cv2.FILLED
    )
    cv2.putText(
        image, label,
        (x1 + 2, label_y - 4),
        font, font_scale,
        (0, 0, 0),
        font_thickness,
        cv2.LINE_AA
    )
    return image


# ─────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────

def print_summary(results_log):
    print("\n" + "=" * 65)
    print("  DETECTION SUMMARY")
    print("=" * 65)
    print(f"  {'Image':<22} {'Class':<12} {'Confidence':>10}  {'Model'}")
    print("-" * 65)

    if not results_log:
        print("  No detections found.")
    else:
        for e in results_log:
            conf = e["cnn_conf"] if e["model_used"] == "CNN" else e["yolo_conf"]
            print(f"  {e['image']:<22} {e['class']:<12} {conf:>9.1%}  [{e['model_used']}]")

    yolo_count = sum(1 for e in results_log if e["model_used"] == "YOLO")
    cnn_count  = sum(1 for e in results_log if e["model_used"] == "CNN")

    print("=" * 65)
    print(f"  Total detections : {len(results_log)}")
    print(f"  YOLO decisions   : {yolo_count}")
    print(f"  CNN overrides    : {cnn_count}")
    print("=" * 65 + "\n")


def get_image_paths(source):
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            return [source]
        else:
            print(f"[ERROR] Unsupported format: {source}")
            sys.exit(1)
    elif source.is_dir():
        paths = [p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
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

def run_detection(source, model_path, cnn_path, output_folder, conf_threshold):
    """
    Hybrid YOLO + CNN detection pipeline:
    1. Load YOLO and CNN models
    2. For each image:
       a. Preprocess with OpenCV
       b. YOLO detects objects and draws bounding boxes
       c. For each detection, crop the ROI
       d. CNN classifies the crop
       e. If CNN confidence >= threshold, CNN overrides YOLO label
       f. Draw final result with model indicator
    3. Save output and print summary
    """

    # Load models
    yolo_model             = load_yolo(model_path)
    cnn_model, cnn_classes = load_cnn(cnn_path)

    hybrid_mode = cnn_model is not None
    print(f"\n[INFO] Mode: {'Hybrid YOLO + CNN' if hybrid_mode else 'YOLO only'}")
    print(f"[INFO] CNN confidence threshold: {CNN_CONFIDENCE_THRESHOLD:.0%}")
    print(f"[INFO] Preprocessing: noise={'ON' if ENABLE_NOISE_REMOVAL else 'OFF'}, "
          f"contrast={'ON' if ENABLE_CONTRAST_ENHANCE else 'OFF'}\n")

    # Prepare output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Get image paths
    image_paths = get_image_paths(source)
    print(f"[INFO] Found {len(image_paths)} image(s) to process.\n")

    results_log = []
    start_time  = time.time()

    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Processing: {image_path.name}")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  [WARNING] Could not read: {image_path.name} — skipping.")
            continue

        h, w = image.shape[:2]

        # Preprocess for YOLO
        image_processed = preprocess_image(image)
        

        # Run YOLO


        results = yolo_model(image_processed, conf=conf_threshold, iou=0.45, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                print(f"  [INFO] No detections.")
                continue

            for box in boxes:
                # YOLO outputs
                yolo_class_id = int(box.cls[0])
                yolo_class    = yolo_model.names[yolo_class_id]
                yolo_conf     = float(box.conf[0])
                coords        = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)

                # Final decision defaults to YOLO
                final_class  = yolo_class
                final_conf   = yolo_conf
                cnn_conf     = 0.0
                model_used   = "YOLO"

                # ── CNN refinement ───────────────
                if hybrid_mode:
                    # Crop the detected region from original image
                    pad   = 5
                    cx1   = max(0, x1 - pad)
                    cy1   = max(0, y1 - pad)
                    cx2   = min(w, x2 + pad)
                    cy2   = min(h, y2 + pad)
                    crop  = image[cy1:cy2, cx1:cx2]

                    if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                        cnn_class, cnn_conf = classify_crop(crop, cnn_model, cnn_classes)

                        # CNN overrides YOLO if confidence is high enough
                        if cnn_conf >= CNN_CONFIDENCE_THRESHOLD:
                            final_class = cnn_class
                            final_conf  = cnn_conf
                            model_used  = "CNN"

                            if cnn_class != yolo_class:
                                print(f"  ⚡ CNN override: {yolo_class} ({yolo_conf:.1%}) "
                                      f"→ {cnn_class} ({cnn_conf:.1%})")

                # Draw on original image
                colour = CLASS_COLOURS.get(final_class, DEFAULT_COLOUR)
                image  = draw_detection(
                    image, coords,
                    final_class, yolo_conf, cnn_conf,
                    model_used, colour
                )

                results_log.append({
                    "image":      image_path.name,
                    "class":      final_class,
                    "yolo_class": yolo_class,
                    "yolo_conf":  yolo_conf,
                    "cnn_conf":   cnn_conf,
                    "model_used": model_used,
                })

                conf_display = cnn_conf if model_used == "CNN" else yolo_conf
                print(f"  ✓ {final_class} ({conf_display:.1%}) [{model_used}]")

        # Save output
        output_path = Path(output_folder) / f"result_{image_path.name}"
        cv2.imwrite(str(output_path), image)
        print(f"  Saved → {output_path}\n")

    elapsed = time.time() - start_time
    print_summary(results_log)
    print(f"[INFO] Done in {elapsed:.1f}s. Results saved to: {output_folder}\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="EcoSort AI — Hybrid YOLO + CNN Waste Detection"
    )
    parser.add_argument("--source",  type=str,   default=INPUT_FOLDER)
    parser.add_argument("--model",   type=str,   default=MODEL_PATH)
    parser.add_argument("--cnn",     type=str,   default=CNN_PATH)
    parser.add_argument("--output",  type=str,   default=OUTPUT_FOLDER)
    parser.add_argument("--conf",    type=float, default=YOLO_CONFIDENCE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_detection(
        source         = args.source,
        model_path     = args.model,
        cnn_path       = args.cnn,
        output_folder  = args.output,
        conf_threshold = args.conf
    )

# =============================================================
#  EcoSort AI — Flask Web Application
#  app.py
#
#  Usage:
#    python app.py
#    Then open http://127.0.0.1:5000 in your browser
# =============================================================

import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

MODEL_PATH      = r"C:\Users\Victus\Documents\GitHub\TrashDetect\runs\detect\ecosort_v3\weights\best.pt"
UPLOAD_FOLDER   = r"C:\Users\Victus\Documents\GitHub\TrashDetect\Test_Image(Input)"
OUTPUT_FOLDER   = r"C:\Users\Victus\Documents\GitHub\TrashDetect\Result(Output)"

CONFIDENCE_THRESHOLD = 0.50

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

# Preprocessing flags
ENABLE_NOISE_REMOVAL    = True
ENABLE_CONTRAST_ENHANCE = True
ENABLE_SHARPEN          = False

# Class colours (BGR for OpenCV)
CLASS_COLOURS = {
    "cardboard": (0,   165, 255),
    "glass":     (208, 224,  64),
    "metal":     (192, 192, 192),
    "paper":     (102, 255, 255),
    "plastic":   (255, 153,  51),
}
DEFAULT_COLOUR = (0, 255, 0)

# Recyclability info per class
CLASS_INFO = {
    "cardboard": {"recyclable": "Yes", "bin": "Blue Bin"},
    "glass":     {"recyclable": "Yes", "bin": "Orange Bin"},
    "metal":     {"recyclable": "Yes", "bin": "Orange Bin"},
    "paper":     {"recyclable": "Yes", "bin": "Blue Bin"},
    "plastic":   {"recyclable": "Yes", "bin": "Orange Bin"},
}

# ─────────────────────────────────────────────
#  FLASK APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

# Create folders if they don't exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Load model once at startup
print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)
print(f"[INFO] Model loaded. Classes: {model.names}")


# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_image(image):
    preprocessed = image.copy()
    if ENABLE_NOISE_REMOVAL:
        preprocessed = cv2.GaussianBlur(preprocessed, (3, 3), 0)
    if ENABLE_CONTRAST_ENHANCE:
        lab   = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        lab   = cv2.merge((l, a, b))
        preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if ENABLE_SHARPEN:
        blurred      = cv2.GaussianBlur(preprocessed, (0, 0), 3)
        preprocessed = cv2.addWeighted(preprocessed, 1.5, blurred, -0.5, 0)
    return preprocessed


# ─────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────

def draw_detection(image, box, class_name, confidence, colour):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
    label = f"{class_name} {confidence:.2f}"
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
#  HELPERS
# ─────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a JPG, PNG, BMP or WEBP image."}), 400

    # Save uploaded image
    ext          = file.filename.rsplit(".", 1)[1].lower()
    unique_name  = f"{uuid.uuid4().hex}.{ext}"
    input_path   = Path(UPLOAD_FOLDER) / unique_name
    file.save(str(input_path))

    # Read image
    image = cv2.imread(str(input_path))
    if image is None:
        return jsonify({"error": "Could not read the uploaded image."}), 400

    # Preprocess
    image_for_detection = preprocess_image(image)

    # Run YOLO
    start_time = time.time()
    results    = model(image_for_detection, conf=CONFIDENCE_THRESHOLD, verbose=False)
    elapsed    = round((time.time() - start_time) * 1000, 1)

    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            class_id   = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = round(float(box.conf[0]) * 100, 1)
            coords     = box.xyxy[0].tolist()
            colour     = CLASS_COLOURS.get(class_name, DEFAULT_COLOUR)
            image      = draw_detection(image, coords, class_name, confidence / 100, colour)
            info       = CLASS_INFO.get(class_name, {"recyclable": "Unknown", "bin": "Unknown"})
            detections.append({
                "class":      class_name,
                "confidence": confidence,
                "recyclable": info["recyclable"],
                "bin":        info["bin"],
            })

    # Save output image
    output_filename = f"result_{unique_name}"
    output_path     = Path(OUTPUT_FOLDER) / output_filename
    cv2.imwrite(str(output_path), image)

    return jsonify({
        "detections":      detections,
        "total":           len(detections),
        "inference_time":  elapsed,
        "output_filename": output_filename,
        "input_filename":  unique_name,
    })


@app.route("/result/<filename>")
def result_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/input/<filename>")
def input_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/save/<filename>")
def save_result(filename):
    """Returns the result image as a downloadable file."""
    return send_from_directory(
        OUTPUT_FOLDER,
        filename,
        as_attachment=True,
        download_name=f"ecosort_result_{filename}"
    )


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

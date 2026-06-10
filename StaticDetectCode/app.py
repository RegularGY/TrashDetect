# =============================================================
#  EcoSort AI — Flask Web Application (Phase 6 Hybrid)
#  app.py
#
#  Usage:
#    python app.py
#    Then open http://127.0.0.1:5000
# =============================================================

import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

YOLO_PATH       = r"C:\Users\Victus\Documents\GitHub\TrashDetect\runs\detect\ecosort_v3\weights\best.pt"
CNN_PATH        = r"C:\Users\Victus\Documents\GitHub\TrashDetect\runs\cnn\ecosort_cnn_v1.pt"
UPLOAD_FOLDER   = r"C:\Users\Victus\Documents\GitHub\TrashDetect\Test_Image(Input)"
OUTPUT_FOLDER   = r"C:\Users\Victus\Documents\GitHub\TrashDetect\Result(Output)"

YOLO_CONFIDENCE      = 0.50
CNN_CONF_THRESHOLD   = 0.70
CNN_SIZE             = 224
ALLOWED_EXTENSIONS   = {"jpg", "jpeg", "png", "bmp", "webp"}

ENABLE_NOISE_REMOVAL    = True
ENABLE_CONTRAST_ENHANCE = True
ENABLE_SHARPEN          = False

CLASS_COLOURS = {
    "cardboard": (0,   165, 255),
    "glass":     (208, 224,  64),
    "metal":     (192, 192, 192),
    "paper":     (102, 255, 255),
    "plastic":   (255, 153,  51),
}
DEFAULT_COLOUR = (0, 255, 0)

CLASS_INFO = {
    "cardboard": {"recyclable": "Yes", "bin": "Blue Bin"},
    "glass":     {"recyclable": "Yes", "bin": "Orange Bin"},
    "metal":     {"recyclable": "Yes", "bin": "Orange Bin"},
    "paper":     {"recyclable": "Yes", "bin": "Blue Bin"},
    "plastic":   {"recyclable": "Yes", "bin": "Orange Bin"},
}

# ─────────────────────────────────────────────
#  FLASK SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOAD MODELS AT STARTUP
# ─────────────────────────────────────────────

print("[INFO] Loading YOLO model...")
yolo_model = YOLO(YOLO_PATH)
print(f"[INFO] YOLO loaded. Classes: {yolo_model.names}")

print("[INFO] Loading CNN model...")
cnn_model   = None
cnn_classes = None

if Path(CNN_PATH).exists():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CNN_PATH, map_location=device)
    cnn_classes = checkpoint["classes"]

    resnet      = models.resnet18(weights=None)
    resnet.fc   = nn.Linear(resnet.fc.in_features, checkpoint["num_classes"])
    resnet.load_state_dict(checkpoint["model_state_dict"])
    resnet      = resnet.to(device)
    resnet.eval()
    cnn_model   = resnet

    print(f"[INFO] CNN loaded. Best val accuracy: {checkpoint.get('best_val_acc', 0):.1%}")
    print(f"[INFO] Running in HYBRID mode (YOLO + CNN)\n")
else:
    print("[WARNING] CNN model not found — running in YOLO-only mode.\n")

# CNN transform
cnn_transform = transforms.Compose([
    transforms.Resize((CNN_SIZE, CNN_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def preprocess_image(image):
    p = image.copy()
    if ENABLE_NOISE_REMOVAL:
        p = cv2.GaussianBlur(p, (3, 3), 0)
    if ENABLE_CONTRAST_ENHANCE:
        lab     = cv2.cvtColor(p, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        p       = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if ENABLE_SHARPEN:
        blurred = cv2.GaussianBlur(p, (0, 0), 3)
        p       = cv2.addWeighted(p, 1.5, blurred, -0.5, 0)
    return p


def classify_crop(crop_bgr):
    if cnn_model is None:
        return None, 0.0
    device  = next(cnn_model.parameters()).device
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor  = cnn_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs              = cnn_model(tensor)
        probs                = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return cnn_classes[predicted.item()], confidence.item()


def draw_detection(image, box, class_name, yolo_conf, cnn_conf, model_used, colour):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
    label = f"{class_name} {(cnn_conf if model_used == 'CNN' else yolo_conf):.2f} [{model_used}]"
    font           = cv2.FONT_HERSHEY_SIMPLEX
    font_scale     = 1.2
    font_thickness = 3
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    label_y = max(y1 - 10, text_h + 10)
    cv2.rectangle(
        image,
        (x1, label_y - text_h - baseline - 4),
        (x1 + text_w + 4, label_y + baseline - 4),
        colour, cv2.FILLED
    )
    cv2.putText(image, label, (x1 + 2, label_y - 4),
                font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    return image


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           hybrid_mode=cnn_model is not None)


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Please upload a JPG, PNG, BMP or WEBP image."}), 400

    # Save uploaded file
    ext         = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    input_path  = Path(UPLOAD_FOLDER) / unique_name
    file.save(str(input_path))

    image = cv2.imread(str(input_path))
    if image is None:
        return jsonify({"error": "Could not read image."}), 400

    h, w       = image.shape[:2]
    processed  = preprocess_image(image)
    start_time = time.time()
    results    = yolo_model(processed, conf=YOLO_CONFIDENCE, verbose=False)
    elapsed    = round((time.time() - start_time) * 1000, 1)

    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            yolo_class_id = int(box.cls[0])
            yolo_class    = yolo_model.names[yolo_class_id]
            yolo_conf     = float(box.conf[0])
            coords        = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)

            final_class = yolo_class
            cnn_conf    = 0.0
            model_used  = "YOLO"

            # CNN refinement
            if cnn_model is not None:
                pad  = 5
                crop = image[max(0, y1-pad):min(h, y2+pad),
                             max(0, x1-pad):min(w, x2+pad)]
                if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                    cnn_class, cnn_conf = classify_crop(crop)
                    if cnn_conf >= CNN_CONF_THRESHOLD:
                        final_class = cnn_class
                        model_used  = "CNN"

            colour = CLASS_COLOURS.get(final_class, DEFAULT_COLOUR)
            image  = draw_detection(image, coords, final_class,
                                    yolo_conf, cnn_conf, model_used, colour)

            info = CLASS_INFO.get(final_class, {"recyclable": "Unknown", "bin": "Unknown"})
            conf_display = round((cnn_conf if model_used == "CNN" else yolo_conf) * 100, 1)

            detections.append({
                "class":       final_class,
                "confidence":  conf_display,
                "yolo_class":  yolo_class,
                "yolo_conf":   round(yolo_conf * 100, 1),
                "cnn_conf":    round(cnn_conf * 100, 1),
                "model_used":  model_used,
                "recyclable":  info["recyclable"],
                "bin":         info["bin"],
            })

    output_filename = f"result_{unique_name}"
    cv2.imwrite(str(Path(OUTPUT_FOLDER) / output_filename), image)

    return jsonify({
        "detections":      detections,
        "total":           len(detections),
        "inference_time":  elapsed,
        "output_filename": output_filename,
        "input_filename":  unique_name,
        "hybrid_mode":     cnn_model is not None,
    })


@app.route("/result/<filename>")
def result_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/input/<filename>")
def input_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/save/<filename>")
def save_result(filename):
    return send_from_directory(OUTPUT_FOLDER, filename,
                               as_attachment=True,
                               download_name=f"ecosort_result_{filename}")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

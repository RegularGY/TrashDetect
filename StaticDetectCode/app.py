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
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, make_response
from ultralytics import YOLO

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table as RLTable,
    TableStyle, Image as RLImage, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

YOLO_PATH            = r"C:\Users\Victus\Documents\GitHub\TrashDetect\runs\detect\ecosort_v5\weights\best.pt"
CNN_PATH             = r"C:\Users\Victus\Documents\GitHub\TrashDetect\runs\cnn\ecosort_cnn_v3.pt"
UPLOAD_FOLDER        = r"C:\Users\Victus\Documents\GitHub\TrashDetect\Test_Image(Input)"
OUTPUT_FOLDER        = r"C:\Users\Victus\Documents\GitHub\TrashDetect\Result(Output)"
CNN_FEEDBACK_FOLDER  = r"C:\Users\Victus\Documents\GitHub\TrashDetect\CNN_Feedback"
YOLO_FEEDBACK_FOLDER = r"C:\Users\Victus\Documents\GitHub\TrashDetect\YOLO_Feedback"
FEEDBACK_LOG         = r"C:\Users\Victus\Documents\GitHub\TrashDetect\feedback.json"

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

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
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Please upload images under 32MB each."}), 413

# ── Feedback dataset folders ──────────────────────────────────
# CNN_Feedback: cropped images sorted by corrected class
for cls in CLASSES:
    Path(CNN_FEEDBACK_FOLDER, cls).mkdir(parents=True, exist_ok=True)

# YOLO_Feedback: full images + auto-generated label files
Path(YOLO_FEEDBACK_FOLDER, 'images').mkdir(parents=True, exist_ok=True)
Path(YOLO_FEEDBACK_FOLDER, 'labels').mkdir(parents=True, exist_ok=True)

# Initialise feedback.json if it doesn't exist
if not Path(FEEDBACK_LOG).exists():
    with open(FEEDBACK_LOG, 'w') as f:
        import json as _json
        _json.dump([], f)

print("[INFO] Feedback folders ready.")
print(f"[INFO]   CNN  feedback → {CNN_FEEDBACK_FOLDER}")
print(f"[INFO]   YOLO feedback → {YOLO_FEEDBACK_FOLDER}")
print(f"[INFO]   Feedback log  → {FEEDBACK_LOG}")

# ─────────────────────────────────────────────
#  IN-MEMORY DETECTION SUMMARY STORE
#  Resets when the Flask server restarts.
#  Each entry: {id, method, image_filename, class, confidence,
#               model_used, recyclable, bin, timestamp}
# ─────────────────────────────────────────────

DETECTION_SUMMARY = []
_summary_id_counter = 0


def add_to_summary(method, output_filename, detections, source_filename=None):
    """Add one image/frame's detections to the global in-memory summary."""
    global _summary_id_counter
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    for det in detections:
        _summary_id_counter += 1
        DETECTION_SUMMARY.append({
            "id":              _summary_id_counter,
            "method":          method,                 # "Static Image" or "Live Camera"
            "source_filename": source_filename or "—",
            "output_filename": output_filename,        # None for live camera unless captured
            "class":           det["class"],
            "confidence":      det["confidence"],
            "model_used":      det["model_used"],
            "recyclable":      det["recyclable"],
            "bin":             det["bin"],
            "timestamp":       timestamp,
        })

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
def landing():
    """Landing page — user chooses Static Image or Live Camera detection."""
    return render_template("landing.html")


@app.route("/static-detection")
def static_detection():
    """Static image detection page (existing upload + detect UI)."""
    return render_template("static_detect.html",
                           hybrid_mode=cnn_model is not None)


@app.route("/live-detection")
def live_detection():
    """Live camera detection page using browser webcam."""
    return render_template("live_detect.html")


@app.route("/detect-frame", methods=["POST"])
def detect_frame():
    """
    Fast YOLO-only detection endpoint for live camera frames.
    CNN is skipped here to maintain real-time frame rate, but a
    lightweight CLAHE contrast enhancement is applied since it adds
    minimal latency (~2-5ms) while improving classification accuracy
    under typical webcam lighting conditions.
    """
    if "frame" not in request.files:
        return jsonify({"error": "No frame received."}), 400

    file = request.files["frame"]
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Could not decode frame."}), 400

    h, w = image.shape[:2]

    start_time = time.time()

    # Lightweight CLAHE only (no Gaussian blur, no CNN) — kept minimal
    # to preserve real-time frame rate while still improving contrast
    # for webcam frames, which are typically lower quality than the
    # dedicated camera used for static image dataset collection.
    lab     = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    frame_for_detection = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    results = yolo_model(frame_for_detection, conf=YOLO_CONFIDENCE, iou=0.45, verbose=False)
    elapsed = round((time.time() - start_time) * 1000, 1)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            class_id   = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            conf       = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Normalise coordinates (0-1) so frontend canvas can scale
            # regardless of the actual frame resolution sent
            detections.append({
                "class":      class_name,
                "confidence": round(conf * 100, 1),
                "box_norm":   [x1 / w, y1 / h, x2 / w, y2 / h],
            })

    return jsonify({
        "detections":     detections,
        "inference_time": elapsed,
    })


@app.route("/capture-frame", methods=["POST"])
def capture_frame():
    """
    Saves a snapshot from the live camera (with CNN refinement applied)
    and adds it to the global Detection Summary.
    Called when the user clicks "Save Current Frame" on the live camera page.
    """
    if "frame" not in request.files:
        return jsonify({"error": "No frame received."}), 400

    file = request.files["frame"]
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Could not decode frame."}), 400

    h, w = image.shape[:2]

    # Run full hybrid pipeline on the captured snapshot (CNN included,
    # since this is a one-off capture, not a continuous live stream)
    processed = preprocess_image(image)
    results   = yolo_model(processed, conf=YOLO_CONFIDENCE, iou=0.45, verbose=False)

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
                "class":      final_class,
                "confidence": conf_display,
                "model_used": model_used,
                "recyclable": info["recyclable"],
                "bin":        info["bin"],
            })

    output_filename = f"result_camera_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(Path(OUTPUT_FOLDER) / output_filename), image)

    add_to_summary("Live Camera", output_filename, detections, source_filename="Webcam Capture")

    return jsonify({
        "detections":      detections,
        "total":           len(detections),
        "output_filename": output_filename,
    })


@app.route("/summary")
def summary():
    """Detection summary page — shows all results from both Static Image and Live Camera."""
    total       = len(DETECTION_SUMMARY)
    static_cnt  = sum(1 for d in DETECTION_SUMMARY if d["method"] == "Static Image")
    camera_cnt  = sum(1 for d in DETECTION_SUMMARY if d["method"] == "Live Camera")
    cnn_cnt     = sum(1 for d in DETECTION_SUMMARY if d["model_used"] == "CNN")
    unique_cls  = len(set(d["class"] for d in DETECTION_SUMMARY))

    response = make_response(render_template(
        "summary.html",
        entries=list(reversed(DETECTION_SUMMARY)),  # newest first
        total=total,
        static_cnt=static_cnt,
        camera_cnt=camera_cnt,
        cnn_cnt=cnn_cnt,
        unique_cls=unique_cls,
    ))
    # Prevent browser from caching this page so corrected state
    # is always reflected correctly when navigating back
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return response


@app.route("/summary/delete/<int:entry_id>", methods=["POST"])
def delete_summary_entry(entry_id):
    """
    Deletes a single entry from the in-memory Detection Summary.
    Used to remove blurry, duplicate, or unwanted detection entries
    before downloading the PDF report.
    Note: this does not affect feedback.json or the feedback dataset
    since deletion only removes the summary entry, not any saved crops.
    """
    global DETECTION_SUMMARY
    original_len = len(DETECTION_SUMMARY)
    DETECTION_SUMMARY = [e for e in DETECTION_SUMMARY if e["id"] != entry_id]

    if len(DETECTION_SUMMARY) < original_len:
        return jsonify({"status": "deleted", "id": entry_id})
    else:
        return jsonify({"error": "Entry not found."}), 404


@app.route("/feedback-log")
def feedback_log():
    """Read-only page showing all corrections made so far."""
    try:
        with open(FEEDBACK_LOG, 'r') as f:
            log = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        log = []

    total_corrections = len(log)
    cnn_saved  = sum(1 for e in log if e.get('cnn_saved'))
    yolo_saved = sum(1 for e in log if e.get('yolo_saved'))
    ready_to_retrain = total_corrections >= 20

    return render_template(
        "feedback_log.html",
        entries=list(reversed(log)),
        total_corrections=total_corrections,
        cnn_saved=cnn_saved,
        yolo_saved=yolo_saved,
        ready_to_retrain=ready_to_retrain,
    )


@app.route("/summary/clear", methods=["POST"])
def clear_summary():
    """Clears ALL entries from the in-memory detection summary and resets the ID counter.
    Works regardless of whether entries are corrected or not."""
    global _summary_id_counter
    DETECTION_SUMMARY.clear()
    _summary_id_counter = 0
    return jsonify({"status": "cleared", "total": 0})


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Handles user category correction from the Detection Summary page.

    For each corrected detection:
    1. Re-crops the object from the saved output image
    2. Saves the crop to CNN_Feedback/[corrected_class]/ for CNN retraining
    3. Saves the full output image + auto-generated YOLO label file
       to YOLO_Feedback/ for YOLO retraining
    4. Logs the correction to feedback.json
    5. Updates the in-memory DETECTION_SUMMARY entry

    Note: YOLO retraining feedback is only generated when the output
    image exists and the original bounding box coordinates are available.
    This covers cases where YOLO correctly localised the object but
    assigned the wrong class label — the most common correction scenario.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received."}), 400

    entry_id       = data.get("id")
    corrected_class = data.get("corrected_class", "").lower().strip()

    if corrected_class not in CLASSES:
        return jsonify({"error": f"Invalid class: {corrected_class}"}), 400

    # Find the entry in DETECTION_SUMMARY
    entry = next((e for e in DETECTION_SUMMARY if e["id"] == entry_id), None)
    if not entry:
        return jsonify({"error": "Detection entry not found."}), 404

    original_class  = entry["class"]
    output_filename = entry["output_filename"]
    method          = entry["method"]
    source_filename = entry["source_filename"]
    confidence      = entry["confidence"]
    model_used      = entry["model_used"]

    # ── Step 1: Load saved output image ──────────────────────────
    output_path = Path(OUTPUT_FOLDER) / output_filename
    if not output_path.exists():
        return jsonify({"error": "Output image not found on disk."}), 404

    image = cv2.imread(str(output_path))
    if image is None:
        return jsonify({"error": "Could not read output image."}), 400

    h, w = image.shape[:2]

    # ── Step 2: Re-run YOLO on the output image to get box coords ─
    # We run YOLO on the original image to get normalised coordinates
    # needed for the YOLO label file. This ensures we have accurate
    # box positions even if the output image has annotations drawn on it.
    yolo_results = yolo_model(image, conf=YOLO_CONFIDENCE, iou=0.45, verbose=False)

    cnn_crop      = None
    box_norm      = None   # [x_center, y_center, width, height] normalised

    for result in yolo_results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        # Pick the highest-confidence detection as the primary object
        best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        # CNN crop (224x224) for CNN_Feedback
        pad  = 5
        cx1  = max(0, x1 - pad)
        cy1  = max(0, y1 - pad)
        cx2  = min(w, x2 + pad)
        cy2  = min(h, y2 + pad)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            cnn_crop = cv2.resize(crop, (CNN_SIZE, CNN_SIZE))

        # YOLO normalised box coordinates
        bx_center = ((x1 + x2) / 2) / w
        by_center = ((y1 + y2) / 2) / h
        bw        = (x2 - x1) / w
        bh        = (y2 - y1) / h
        box_norm  = [bx_center, by_center, bw, bh]
        break

    uid = uuid.uuid4().hex[:10]

    # ── Step 3: Save CNN crop ─────────────────────────────────────
    cnn_saved = False
    if cnn_crop is not None:
        cnn_path = Path(CNN_FEEDBACK_FOLDER) / corrected_class / f"{corrected_class}_{uid}.jpg"
        cv2.imwrite(str(cnn_path), cnn_crop)
        cnn_saved = True

    # ── Step 4: Save YOLO full image + label file ─────────────────
    yolo_saved = False
    if box_norm is not None:
        class_id   = CLASSES.index(corrected_class)
        label_line = f"{class_id} {box_norm[0]:.6f} {box_norm[1]:.6f} {box_norm[2]:.6f} {box_norm[3]:.6f}"

        yolo_img_path   = Path(YOLO_FEEDBACK_FOLDER) / 'images' / f"{uid}.jpg"
        yolo_label_path = Path(YOLO_FEEDBACK_FOLDER) / 'labels' / f"{uid}.txt"

        cv2.imwrite(str(yolo_img_path), image)
        yolo_label_path.write_text(label_line)
        yolo_saved = True

    # ── Step 5: Log to feedback.json ──────────────────────────────
    feedback_entry = {
        "id":               uid,
        "summary_entry_id": entry_id,
        "method":           method,
        "source_filename":  source_filename,
        "output_filename":  output_filename,
        "original_class":   original_class,
        "corrected_class":  corrected_class,
        "confidence":       confidence,
        "model_used":       model_used,
        "cnn_saved":        cnn_saved,
        "yolo_saved":       yolo_saved,
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with open(FEEDBACK_LOG, 'r') as f:
            log = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        log = []

    log.append(feedback_entry)

    with open(FEEDBACK_LOG, 'w') as f:
        json.dump(log, f, indent=2)

    # ── Step 6: Update in-memory Detection Summary ────────────────
    info = CLASS_INFO.get(corrected_class, {"recyclable": "Unknown", "bin": "Unknown"})
    entry["class"]      = corrected_class
    entry["recyclable"] = info["recyclable"]
    entry["bin"]        = info["bin"]
    entry["corrected"]  = True

    print(f"[FEEDBACK] #{entry_id} corrected: {original_class} → {corrected_class} "
          f"| CNN saved: {cnn_saved} | YOLO saved: {yolo_saved}")

    return jsonify({
        "status":          "saved",
        "original_class":  original_class,
        "corrected_class": corrected_class,
        "recyclable":      info["recyclable"],
        "bin":             info["bin"],
        "cnn_saved":       cnn_saved,
        "yolo_saved":      yolo_saved,
    })


@app.route("/summary/download")
def download_summary():
    """
    Generates a combined PDF report of all detection results,
    sectioned by method (Static Image / Live Camera).
    """
    if not DETECTION_SUMMARY:
        return "<h2 style='font-family:sans-serif;text-align:center;margin-top:100px;'>" \
               "No detections to export yet.</h2>" \
               "<p style='text-align:center;'><a href='/summary'>← Back to summary</a></p>"

    pdf_path = Path(OUTPUT_FOLDER) / f"EcoSort_Detection_Report_{uuid.uuid4().hex[:8]}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        topMargin=18*mm, bottomMargin=16*mm,
        leftMargin=18*mm, rightMargin=18*mm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleC', parent=styles['Title'], fontSize=20,
                                  textColor=colors.HexColor('#15803d'), alignment=TA_CENTER, spaceAfter=4)
    subtitle_style = ParagraphStyle('SubtitleC', parent=styles['Normal'], fontSize=10.5,
                                     textColor=colors.HexColor('#667085'), alignment=TA_CENTER, spaceAfter=2)
    section_style = ParagraphStyle('SectionH', parent=styles['Heading1'], fontSize=14,
                                    textColor=colors.HexColor('#101828'), spaceBefore=14, spaceAfter=8)
    entry_title_style = ParagraphStyle('EntryTitle', parent=styles['Heading2'], fontSize=11,
                                        textColor=colors.HexColor('#101828'), spaceBefore=10, spaceAfter=4)
    label_style = ParagraphStyle('Label', parent=styles['Normal'], fontSize=9.5,
                                  textColor=colors.HexColor('#344054'), leading=14)

    story = []

    # ── Cover Section ─────────────────────────────────────────
    story.append(Paragraph("♻️ EcoSort AI", title_style))
    story.append(Paragraph("Detection Report", subtitle_style))
    story.append(Spacer(1, 10*mm))

    total      = len(DETECTION_SUMMARY)
    static_cnt = sum(1 for d in DETECTION_SUMMARY if d["method"] == "Static Image")
    camera_cnt = sum(1 for d in DETECTION_SUMMARY if d["method"] == "Live Camera")
    cnn_cnt    = sum(1 for d in DETECTION_SUMMARY if d["model_used"] == "CNN")
    generated  = time.strftime("%d %B %Y, %H:%M:%S")

    cover_data = [
        ["Report Generated:", generated],
        ["Total Detections:", str(total)],
        ["Static Image Detections:", str(static_cnt)],
        ["Live Camera Detections:", str(camera_cnt)],
        ["CNN Overrides:", str(cnn_cnt)],
    ]
    cover_table = RLTable(cover_data, colWidths=[60*mm, 80*mm])
    cover_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#344054')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#101828')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor('#e4e7ec')),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#16a34a')))
    story.append(PageBreak())

    # ── Helper to render one section (Static Image or Live Camera) ──
    def render_section(method_name, icon_label):
        # Group entries by output_filename (each image/frame may have
        # multiple detections sharing the same annotated output image)
        method_entries = [e for e in DETECTION_SUMMARY if e["method"] == method_name]
        if not method_entries:
            return False

        story.append(Paragraph(f"{icon_label} {method_name} Detections", section_style))
        story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor('#e4e7ec')))

        grouped = {}
        order = []
        for e in method_entries:
            key = e["output_filename"]
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(e)

        for key in order:
            group = grouped[key]
            first = group[0]

            entry_flowables = []

            entry_flowables.append(Paragraph(
                f"Source: {first['source_filename']}  &nbsp;&nbsp;|&nbsp;&nbsp;  Timestamp: {first['timestamp']}",
                entry_title_style
            ))

            img_path = Path(OUTPUT_FOLDER) / key
            row_cells = []
            if img_path.exists():
                try:
                    rl_img = RLImage(str(img_path), width=70*mm, height=52*mm)
                    rl_img.hAlign = 'LEFT'
                    row_cells.append(rl_img)
                except Exception:
                    row_cells.append(Paragraph("[Image unavailable]", label_style))
            else:
                row_cells.append(Paragraph("[Image not found]", label_style))

            # Build a small table of detections for this image
            det_rows = [["Class", "Confidence", "Model", "Recyclable", "Bin"]]
            for e in group:
                det_rows.append([
                    e["class"].capitalize(),
                    f"{e['confidence']}%",
                    e["model_used"],
                    e["recyclable"],
                    e["bin"],
                ])
            det_table = RLTable(det_rows, colWidths=[24*mm, 22*mm, 16*mm, 20*mm, 22*mm])
            det_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f7f8fa')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#e4e7ec')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))

            combined = RLTable([[row_cells[0], det_table]], colWidths=[74*mm, 104*mm])
            combined.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (0, 0), 0),
            ]))
            entry_flowables.append(combined)
            entry_flowables.append(Spacer(1, 6*mm))

            # Keep the source label, image and table together so a page
            # break never lands in the middle of one detection entry
            story.append(KeepTogether(entry_flowables))

        return True

    # ── Section 1: Static Image ──────────────────────────────────
    has_static = render_section("Static Image", "📤")
    if has_static and camera_cnt > 0:
        story.append(PageBreak())

    # ── Section 2: Live Camera ───────────────────────────────────
    render_section("Live Camera", "🎥")

    doc.build(story)

    return send_file(str(pdf_path), as_attachment=True,
                      download_name="EcoSort_Detection_Report.pdf")


@app.route("/detect", methods=["POST"])
def detect():
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded."}), 400

    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected."}), 400

    all_results = []

    for file in files:
        if file.filename == "" or not allowed_file(file.filename):
            continue

        # Save uploaded file
        ext         = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        input_path  = Path(UPLOAD_FOLDER) / unique_name
        file.save(str(input_path))

        image = cv2.imread(str(input_path))
        if image is None:
            continue

        h, w       = image.shape[:2]
        processed  = preprocess_image(image)
        start_time = time.time()
        results    = yolo_model(processed, conf=YOLO_CONFIDENCE, iou=0.45, verbose=False)
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

        # Add to global detection summary (Static Image method)
        add_to_summary("Static Image", output_filename, detections, source_filename=file.filename)

        all_results.append({
            "detections":      detections,
            "total":           len(detections),
            "inference_time":  elapsed,
            "output_filename": output_filename,
            "input_filename":  file.filename,
        })

    if not all_results:
        return jsonify({"error": "No valid images were processed. Please upload JPG, PNG, BMP or WEBP files."}), 400

    return jsonify({
        "results":     all_results,
        "hybrid_mode": cnn_model is not None,
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

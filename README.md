# EcoSort AI — Trash Detection and Classification System

An AI-powered waste detection and classification system built with a hybrid YOLOv8 + ResNet18 CNN architecture. The system detects trash in images, draws bounding boxes around detected objects, and displays the predicted category with a confidence percentage. A web-based interface built with Flask and HTML allows users to upload images and view detection results without using the terminal.

## Project Overview

This project uses a phased dataset strategy combined with a hybrid YOLO + CNN architecture to detect and classify recyclable waste materials. YOLOv8 handles object localisation and detection, while ResNet18 CNN refines the classification on each cropped detection. Each dataset phase introduces a single new variable — object count, background complexity, or contamination level — to progressively improve model robustness.

### Classes
- Cardboard
- Glass
- Metal
- Paper
- Plastic

### Development Phases
| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Clean single objects, controlled background | ✅ Complete |
| Phase 2 | Clean multiple objects, controlled background | ✅ Complete |
| Phase 2.5 | Clean single objects, varied backgrounds (grass, concrete, road) | ✅ Complete |
| Phase 3 | Dirty single objects, varied backgrounds | ✅ Complete |
| Phase 4 | Dirty multiple objects, varied backgrounds | ✅ Complete |
| Phase 5 | Real world naturally occurring trash | ✅ Complete |
| Phase 5.5 | HTML + Flask web UI | ✅ Complete |
| Phase 6 | YOLO + CNN hybrid | ✅ Complete |
| Phase 7 | Final testing and evaluation | 🔄 In Progress |

---

## How the Hybrid System Works

```
Input Image
     ↓
OpenCV Preprocessing (noise removal + contrast enhancement)
     ↓
YOLOv8 Detection — finds WHERE the object is + initial class label
     ↓
Crop each detected region (Region of Interest)
     ↓
ResNet18 CNN — classifies the cropped object
     ↓
Decision Logic:
  If CNN confidence ≥ 70% → use CNN label  [CNN]
  Otherwise              → keep YOLO label [YOLO]
     ↓
Draw bounding box + label + confidence + model indicator
     ↓
Save to Result(Output)/
```

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on GTX 1650 4GB)
- Minimum 8GB RAM
- Windows 10/11

### Software
- Python 3.12
- CUDA Toolkit 12.1
- Git

---

## Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/RegularGY/TrashDetect.git
cd TrashDetect
```

### Step 2 — Create and activate virtual environment
```bash
cd StaticDetectCode
python -m venv ecosort-env
ecosort-env\Scripts\activate
```

### Step 3 — Install PyTorch with CUDA support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4 — Install remaining dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Verify GPU is detected
```python
python -c "import torch; print(torch.cuda.is_available())"
```
Should print `True`.

---

## Dataset

The dataset is not included in this repository due to size. To get the dataset:

1. Go to the Roboflow project: [TrashDetect on Roboflow](https://universe.roboflow.com/manass-workspace-vy2gz/trashdetect-s3bsb)
2. Download in **YOLOv8 format**
3. Extract and place contents into the `Dataset/` folder:

```
Dataset/
├── train/
│   └── images/
├── valid/
│   └── images/
├── test/
│   └── images/
└── data.yaml
```

4. Update the paths in `data.yaml` to match your local machine.

---

## Folder Structure

```
TrashDetect/
├── run.bat                        ← double click to launch the web UI
├── StaticDetectCode/
│   ├── app.py                     ← Flask web server (hybrid mode)
│   ├── detect.py                  ← command-line inference script (hybrid mode)
│   ├── generate_crops.py          ← generates CNN training crops from YOLO
│   ├── train_cnn.py               ← trains ResNet18 CNN classifier
│   ├── requirements.txt           ← package dependencies
│   ├── templates/
│   │   └── index.html             ← web UI
│   └── ecosort-env/               ← virtual environment (not pushed)
├── Dataset/
│   ├── data.yaml                  ← dataset config
│   ├── train/                     ← training images (not pushed)
│   ├── valid/                     ← validation images (not pushed)
│   └── test/                      ← test images (not pushed)
├── CNN_Dataset/                   ← cropped CNN training images (not pushed)
├── Test_Image(Input)/             ← place test images here (CLI mode)
├── Result(Output)/                ← detection results saved here
├── .gitignore
└── README.md
```

---

## Running the System

### Option A — Web UI (recommended)

Simply double click `run.bat` in the root folder. It will:
1. Activate the virtual environment automatically
2. Start the Flask server
3. Open the browser at `http://127.0.0.1:5000`

From the web UI you can:
- Upload any image by drag and drop or file browser
- Click **Detect Waste** to run detection
- View the annotated result image with bounding boxes
- See a detection summary table showing class, confidence, which model decided (YOLO or CNN), recyclability and bin colour
- Track CNN override count — how many detections were refined by CNN
- Click **Save Result** to download the annotated image

### Option B — Command Line

Activate your environment first:
```bash
ecosort-env\Scripts\activate
```

Default — processes all images in `Test_Image(Input)/`:
```bash
python detect.py
```

Single image:
```bash
python detect.py --source "path/to/image.jpg"
```

Custom confidence threshold:
```bash
python detect.py --conf 0.70
```

Custom model weights:
```bash
python detect.py --model "path/to/best.pt" --cnn "path/to/cnn.pt"
```

Results are saved to `Result(Output)/` with bounding boxes, class labels, confidence scores, and model indicators overlaid.

---

## Training

### YOLO Training

Train from scratch:
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=416 batch=2 workers=2 name=ecosort_v1
```

Continue from existing weights:
```bash
yolo detect train data=data.yaml model=runs/detect/ecosort_v2/weights/best.pt epochs=50 imgsz=416 batch=2 workers=2 name=ecosort_v3
```

### CNN Training

Step 1 — Generate cropped training images from YOLO detections:
```bash
python generate_crops.py
```

Step 2 — Train ResNet18 CNN on the crops:
```bash
python train_cnn.py
```

CNN model weights are saved to:
```
runs/cnn/ecosort_cnn_v1.pt
```

---

## Image Preprocessing

The inference pipeline applies the following OpenCV preprocessing steps before passing images to the model:

| Step | Method | Purpose |
|------|--------|---------|
| Noise Removal | Gaussian Blur (3×3) | Reduces sensor noise and compression artefacts |
| Contrast Enhancement | CLAHE (clipLimit=1.0) | Improves object visibility under poor lighting |

Detection is run on the preprocessed image but bounding boxes are drawn on the original image so output photos look natural.

---

## Model Performance

### YOLO — ecosort_v2 (Phase 1 only, 2,386 images)

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| All | 0.912 | 0.859 | 0.924 |
| Cardboard | 0.919 | 0.715 | 0.814 |
| Glass | 0.864 | 0.900 | 0.941 |
| Metal | 0.879 | 0.914 | 0.948 |
| Paper | 0.965 | 0.887 | 0.956 |
| Plastic | 0.933 | 0.876 | 0.963 |

### YOLO — ecosort_v3 (All phases, 2,931 images)

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| All | 0.925 | 0.816 | 0.911 |
| Cardboard | 0.925 | 0.709 | 0.841 |
| Glass | 0.914 | 0.795 | 0.907 |
| Metal | 0.929 | 0.884 | 0.946 |
| Paper | 0.935 | 0.870 | 0.938 |
| Plastic | 0.921 | 0.820 | 0.925 |

### CNN — ecosort_cnn_v1 (ResNet18, 3,779 crops)

| Metric | Value |
|--------|-------|
| Architecture | ResNet18 (pretrained ImageNet) |
| Training images | 3,024 |
| Validation images | 755 |
| Best validation accuracy | **93.0%** |
| Training time | 16.9 minutes |
| CNN override threshold | 70% confidence |

---

## Technologies Used

- [YOLOv8](https://github.com/ultralytics/ultralytics) — object detection and localisation
- [ResNet18](https://pytorch.org/vision/stable/models/resnet.html) — CNN classification refinement
- [PyTorch](https://pytorch.org/) — deep learning framework
- [OpenCV](https://opencv.org/) — image preprocessing and bounding box rendering
- [Flask](https://flask.palletsprojects.com/) — web server for UI
- [Roboflow](https://roboflow.com/) — dataset annotation and management
- [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) — GPU acceleration

---

## References

This project is developed as part of a final year project at UiTM, addressing waste management challenges in Malaysia.

---



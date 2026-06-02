# EcoSort AI — Trash Detection and Classification System

An AI-powered waste detection and classification system built with YOLOv8 and Python. The system detects trash in images, draws bounding boxes around detected objects, and displays the predicted category with a confidence percentage.

## Project Overview

This project uses a hybrid YOLO + CNN architecture to detect and classify recyclable waste materials. Phase 1 focuses on single object detection in controlled environments. Future phases will expand to multiple objects, dirty/contaminated waste, and real-world environments.

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
| Phase 2 | Clean multiple objects, controlled background | 🔄 In Progress |
| Phase 2.5 | Clean single objects, varied backgrounds | ⏳ Planned |
| Phase 3a | Dirty single objects, controlled background | ⏳ Planned |
| Phase 3b | Dirty single objects, varied backgrounds | ⏳ Planned |
| Phase 4a | Dirty multiple objects, controlled background | ⏳ Planned |
| Phase 4b | Dirty multiple objects, varied backgrounds | ⏳ Planned |
| Phase 5 | Real world images | ⏳ Planned |
| Phase 5.5 | HTML + Flask UI | ⏳ Planned |
| Phase 6 | YOLO + CNN hybrid | ⏳ Planned |

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
├── StaticDetectCode/
│   ├── detect.py              ← inference script
│   ├── requirements.txt       ← package dependencies
│   └── ecosort-env/           ← virtual environment (not pushed)
├── Dataset/
│   ├── data.yaml              ← dataset config
│   ├── train/                 ← training images (not pushed)
│   ├── valid/                 ← validation images (not pushed)
│   └── test/                  ← test images (not pushed)
├── Test_Image(Input)/         ← place test images here
├── Result(Output)/            ← detection results saved here
├── .gitignore
└── README.md
```

---

## Training

To train the model on your dataset:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=416 batch=2 workers=2 name=ecosort_v1
```

Trained model weights are saved to:
```
runs/detect/ecosort_v1/weights/best.pt
```

---

## Running Detection

### Default — processes all images in Test_Image(Input)/
```bash
python detect.py
```

### Single image
```bash
python detect.py --source "path/to/image.jpg"
```

### Custom confidence threshold
```bash
python detect.py --conf 0.70
```

### Custom model weights
```bash
python detect.py --model "path/to/best.pt"
```

Results are saved to `Result(Output)/` with bounding boxes, class labels, and confidence scores overlaid.

---

## Model Performance (Phase 1 — ecosort_v2)

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| All | 0.912 | 0.859 | 0.924 |
| Cardboard | 0.919 | 0.715 | 0.814 |
| Glass | 0.864 | 0.900 | 0.941 |
| Metal | 0.879 | 0.914 | 0.948 |
| Paper | 0.965 | 0.887 | 0.956 |
| Plastic | 0.933 | 0.876 | 0.963 |

---

## Technologies Used

- [YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [PyTorch](https://pytorch.org/) — deep learning framework
- [OpenCV](https://opencv.org/) — image processing
- [Roboflow](https://roboflow.com/) — dataset annotation and management
- [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) — GPU acceleration

---

## References

This project is developed as part of a final year project at UiTM, addressing waste management challenges in Malaysia aligned with UN SDG 11 and 12.

---

## License

This project is for academic purposes only.

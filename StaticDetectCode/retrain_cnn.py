# =============================================================
#  EcoSort AI — CNN Retraining Script
#  retrain_cnn.py
# =============================================================

import os
import sys
import time
import copy
import json
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, models, transforms

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent.parent
ORIGINAL_CROPS = BASE_DIR / "CNN_Dataset"
FEEDBACK_CROPS = BASE_DIR / "CNN_Feedback"
OUTPUT_DIR     = BASE_DIR / "runs" / "cnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_LOG   = BASE_DIR / "feedback.json"

existing = sorted(OUTPUT_DIR.glob("ecosort_cnn_v*.pt"))
if existing:
    last_ver = int(existing[-1].stem.split('_v')[-1])
    new_ver  = last_ver + 1
else:
    new_ver  = 2
CNN_OUTPUT_PATH = OUTPUT_DIR / f"ecosort_cnn_v{new_ver}.pt"

NUM_EPOCHS       = 30
BATCH_SIZE       = 32
LEARNING_RATE    = 0.0005
VALIDATION_SPLIT = 0.2
CNN_SIZE         = 224
CLASSES          = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def retrain_cnn():

    print(f"\n{'=' * 60}")
    print(f"  EcoSort AI — CNN Retraining (ecosort_cnn_v{new_ver})")
    print(f"{'=' * 60}")

    if not ORIGINAL_CROPS.exists():
        print(f"\n[ERROR] Original CNN_Dataset not found at: {ORIGINAL_CROPS}")
        sys.exit(1)

    if not FEEDBACK_CROPS.exists():
        print(f"\n[ERROR] CNN_Feedback folder not found at: {FEEDBACK_CROPS}")
        sys.exit(1)

    feedback_count = sum(
        1 for cls in CLASSES
        for _ in (FEEDBACK_CROPS / cls).glob("*.jpg")
        if (FEEDBACK_CROPS / cls).exists()
    )

    if feedback_count == 0:
        print(f"\n[WARNING] No feedback crops found. Retraining on original dataset only.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # ── Transforms ───────────────────────────────────────────────
    # Apply transforms at load time — NOT after split —
    # because ConcatDataset has no single .dataset.transform attribute
    train_transform = transforms.Compose([
        transforms.Resize((CNN_SIZE, CNN_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CNN_SIZE, CNN_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ── Load datasets — with transforms applied at load time ──────
    print(f"\n[INFO] Loading original dataset: {ORIGINAL_CROPS}")
    original_train = datasets.ImageFolder(str(ORIGINAL_CROPS), transform=train_transform)
    original_val   = datasets.ImageFolder(str(ORIGINAL_CROPS), transform=val_transform)
    print(f"[INFO] Original crops: {len(original_train)}")

    temp_dir = None

    if feedback_count > 0:
        print(f"[INFO] Loading feedback dataset: {FEEDBACK_CROPS}")

        non_empty = [cls for cls in CLASSES
                     if (FEEDBACK_CROPS / cls).exists()
                     and any((FEEDBACK_CROPS / cls).glob("*.jpg"))]
        print(f"[INFO] Feedback classes with images: {non_empty}")

        # Pad any empty class folders with one placeholder image
        # so ImageFolder doesn't crash on empty class directories
        if len(non_empty) < len(CLASSES):
            temp_dir = Path(tempfile.mkdtemp())
            print(f"[INFO] Padding missing feedback classes with placeholder crops...")
            for cls in CLASSES:
                src = FEEDBACK_CROPS / cls
                dst = temp_dir / cls
                dst.mkdir(parents=True, exist_ok=True)
                if src.exists() and any(src.glob("*.jpg")):
                    for img in src.glob("*.jpg"):
                        shutil.copy2(img, dst / img.name)
                else:
                    orig_cls = ORIGINAL_CROPS / cls
                    if orig_cls.exists():
                        first_img = next(orig_cls.glob("*.jpg"), None)
                        if first_img:
                            shutil.copy2(first_img, dst / f"_placeholder_{first_img.name}")
            feedback_src = str(temp_dir)
        else:
            feedback_src = str(FEEDBACK_CROPS)

        feedback_train = datasets.ImageFolder(feedback_src, transform=train_transform)
        feedback_val   = datasets.ImageFolder(feedback_src, transform=val_transform)
        print(f"[INFO] Feedback crops (real): {feedback_count}")

        combined_train = ConcatDataset([original_train, feedback_train])
        combined_val   = ConcatDataset([original_val,   feedback_val])
        print(f"[INFO] Combined total: {len(combined_train)}")
    else:
        combined_train = original_train
        combined_val   = original_val

    # ── Train/val split ──────────────────────────────────────────
    # Use same random seed on both combined_train and combined_val
    # so the split indices are consistent between the two
    total      = len(combined_train)
    val_size   = int(total * VALIDATION_SPLIT)
    train_size = total - val_size

    train_dataset, _ = random_split(
        combined_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    _, val_dataset = random_split(
        combined_val, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\n[INFO] Training images  : {train_size}")
    print(f"[INFO] Validation images: {val_size}")

    # ── Build model ───────────────────────────────────────────────
    existing_weights = sorted(OUTPUT_DIR.glob("ecosort_cnn_v*.pt"))
    if existing_weights:
        prev_path  = existing_weights[-1]
        print(f"\n[INFO] Fine-tuning from: {prev_path.name}")
        checkpoint = torch.load(str(prev_path), map_location=device, weights_only=False)
        model      = models.resnet18(weights=None)
        model.fc   = nn.Linear(model.fc.in_features, len(CLASSES))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"\n[INFO] No prior weights found — using ImageNet pretrained weights")
        model    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    model = model.to(device)

    # ── Loss, optimiser, scheduler ───────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── Training loop ────────────────────────────────────────────
    best_weights = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    start_time   = time.time()

    print(f"\n{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Val Loss':<14} {'Val Acc':<12}")
    print("-" * 62)

    for epoch in range(NUM_EPOCHS):

        model.train()
        train_loss    = 0.0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * inputs.size(0)
            _, preds       = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

        train_loss /= train_size
        train_acc   = train_correct / train_size

        model.eval()
        val_loss    = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs        = model(inputs)
                loss           = criterion(outputs, labels)
                val_loss      += loss.item() * inputs.size(0)
                _, preds       = torch.max(outputs, 1)
                val_correct   += (preds == labels).sum().item()

        val_loss /= val_size
        val_acc   = val_correct / val_size
        scheduler.step()

        marker = " ← best" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        print(f"{epoch+1:<8} {train_loss:<14.4f} {train_acc:<14.1%} {val_loss:<14.4f} {val_acc:<12.1%}{marker}")

    # ── Cleanup temp folder ───────────────────────────────────────
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n[INFO] Temp folder cleaned up.")

    # ── Save ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    model.load_state_dict(best_weights)

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes":          CLASSES,
        "num_classes":      len(CLASSES),
        "cnn_size":         CNN_SIZE,
        "best_val_acc":     best_val_acc,
        "feedback_crops":   feedback_count,
        "version":          new_ver,
    }, str(CNN_OUTPUT_PATH))

    print(f"\n{'=' * 60}")
    print(f"  CNN RETRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Version              : ecosort_cnn_v{new_ver}")
    print(f"  Best val accuracy    : {best_val_acc:.1%}")
    print(f"  Training time        : {elapsed/60:.1f} minutes")
    print(f"  Feedback crops used  : {feedback_count}")
    print(f"  Model saved to       : {CNN_OUTPUT_PATH}")
    print(f"{'=' * 60}")
    print(f"\n[NEXT STEP] Update CNN_PATH in app.py:")
    print(f"  CNN_PATH = r\"{CNN_OUTPUT_PATH}\"")
    print(f"  Then restart Flask.\n")


if __name__ == "__main__":
    retrain_cnn()

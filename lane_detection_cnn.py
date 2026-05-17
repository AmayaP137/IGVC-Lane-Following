# -*- coding: utf-8 -*-
"""
Lane Detection CNN — TuSimple Dataset
======================================
Pixel-level lane segmentation using TuSimple ground-truth seg_label masks.

Output classes:
  0 = Background
  1 = Lane marking (any lane — TuSimple GT does not distinguish white/yellow)

The detector_node uses HSV at runtime to separate white vs yellow after
the CNN finds the lane pixels, so training only needs binary lane/no-lane.

Designed for deployment on Raspberry Pi 5 with ROS 2 Jazzy.
Export target: ONNX for efficient ROS node inference.

Directory structure expected:
  <BASE_DIR>/
    clips_flat/
      train/   ← flattened JPGs  e.g. 0313-1_6040_20.jpg
      val/
      test/
    masks_flat/
      train/   ← generated once by flatten_masks()
      val/
      test/
    train_set/
      seg_label/
        0313-1/  0313-2/  0530/  0531/  0601/
          <clip_id>/
            20.png   ← GT mask, RGB, values 0/2/3/4/5
      label_data_0313.json
      label_data_0531.json
      label_data_0601.json
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as TF

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path("/home/amaya28/ros2_ws/src/archive/TUSimple")
MASK_FLAT_DIR  = BASE_DIR / "masks_flat"
CHECKPOINT_DIR = Path("/home/amaya28/ros2_ws/src/checkpoints")
ONNX_PATH      = Path("/home/amaya28/ros2_ws/src/lane_detector.onnx")

IMG_H, IMG_W   = 256, 512       # model input size (scaled down from 720×1280)
NUM_CLASSES    = 2               # 0 = background, 1 = lane
BATCH_SIZE     = 8
EPOCHS         = 40
LR             = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Using device: {DEVICE}")


# ── Step 1: Flatten GT masks to match clips_flat naming ───────────────────────
# seg_label structure:  seg_label/<date>/<clip_id>/20.png
# clips_flat naming:    <date>_<clip_id>_20.jpg
# mask flat naming:     <date>_<clip_id>_20.png   ← must match image stem
#
# Run once — skip if masks_flat already populated.

def flatten_masks(seg_label_root: Path, out_dir: Path):
    """
    Walk seg_label/, convert each RGB mask to a binary (0/1) single-channel
    PNG, and save with a flattened filename matching clips_flat convention.

    GT mask pixel values:
      0          → background
      2, 3, 4, 5 → lane line indices  →  remapped to 1
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for date_dir in sorted(seg_label_root.iterdir()):
        if not date_dir.is_dir():
            continue
        date = date_dir.name                        # e.g. "0313-1"

        for clip_dir in sorted(date_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            clip_id = clip_dir.name                 # e.g. "6040"

            for mask_png in sorted(clip_dir.glob("*.png")):
                frame = mask_png.stem               # e.g. "20"

                # Load RGB mask, take first channel (R == G == B for these files)
                raw   = np.array(Image.open(mask_png))
                if raw.ndim == 3:
                    raw = raw[:, :, 0]

                # Remap: any non-zero lane index → 1
                binary = (raw > 0).astype(np.uint8)

                flat_name = f"{date}_{clip_id}_{frame}.png"
                Image.fromarray(binary).save(out_dir / flat_name)
                written += 1

    print(f"Flattened {written} masks → {out_dir}")


# ── Run mask flattening once ───────────────────────────────────────────────────
# Covers all dates in seg_label for train split.
# Val masks are carved from the same pool (images were moved with shuf earlier).
# Test set has no seg_label GT — val is used for test evaluation instead.

seg_label_root = BASE_DIR / "train_set" / "seg_label"

train_mask_dir = MASK_FLAT_DIR / "train"
val_mask_dir   = MASK_FLAT_DIR / "val"
test_mask_dir  = MASK_FLAT_DIR / "val"   # reuse val masks for test

if not train_mask_dir.exists() or not any(train_mask_dir.iterdir()):
    print("Flattening GT masks (runs once)...")
    # Flatten all GT masks into a single pool first
    pool_dir = MASK_FLAT_DIR / "_pool"
    flatten_masks(seg_label_root, pool_dir)

    # Split pool to match how clips_flat/train and clips_flat/val were split:
    # move masks whose image stem exists in clips_flat/val → masks_flat/val
    val_mask_dir.mkdir(parents=True, exist_ok=True)
    train_mask_dir.mkdir(parents=True, exist_ok=True)

    val_stems = {
        p.stem
        for p in (BASE_DIR / "clips_flat" / "val").glob("*.jpg")
    }

    for mask_path in pool_dir.glob("*.png"):
        if mask_path.stem in val_stems:
            mask_path.rename(val_mask_dir / mask_path.name)
        else:
            mask_path.rename(train_mask_dir / mask_path.name)

    pool_dir.rmdir()
    print("Mask split complete.")
else:
    print("Masks already flattened — skipping.")


# ── Step 2: Dataset ────────────────────────────────────────────────────────────

class LaneDataset(Dataset):
    """
    Paired (image, binary mask) dataset.
    Only loads samples that have a matching mask file — skips unmatched images.

    images_dir/  *.jpg   →   masks_dir/  *.png  (same stem)
    """

    def __init__(self, images_dir: Path, masks_dir: Path, augment: bool = False):
        self.masks_dir = masks_dir
        self.augment   = augment

        # Only keep images that have a corresponding mask
        all_images = sorted(images_dir.glob("*.jpg"))
        self.pairs = [
            (img, masks_dir / img.with_suffix(".png").name)
            for img in all_images
            if (masks_dir / img.with_suffix(".png").name).exists()
        ]

        assert len(self.pairs) > 0, (
            f"No matched image/mask pairs found.\n"
            f"  images: {images_dir}\n"
            f"  masks:  {masks_dir}\n"
            "Check that flatten_masks() ran correctly."
        )
        print(f"  {images_dir.name}: {len(self.pairs)} paired samples")

        self.normalise = v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB").resize(
            (IMG_W, IMG_H), Image.BILINEAR)
        mask  = Image.open(mask_path).resize(
            (IMG_W, IMG_H), Image.NEAREST)

        image = TF.to_tensor(image)                          # (3,H,W) float32
        mask  = torch.from_numpy(np.array(mask)).long()      # (H,W)   int64  0/1

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # Colour jitter (image only)
            image = v2.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            )(image)

            # Random crop → resize back
            if random.random() > 0.4:
                i, j, h, w = v2.RandomCrop.get_params(
                    image,
                    output_size=(int(IMG_H * 0.85), int(IMG_W * 0.85))
                )
                image = TF.resized_crop(image, i, j, h, w, (IMG_H, IMG_W))
                mask  = TF.resized_crop(
                    mask.unsqueeze(0), i, j, h, w, (IMG_H, IMG_W),
                    interpolation=TF.InterpolationMode.NEAREST
                ).squeeze(0)

        image = self.normalise(image)
        return image, mask


def make_dataloaders():
    print("Building datasets...")
    train_ds = LaneDataset(
        BASE_DIR / "clips_flat" / "train", train_mask_dir, augment=True)
    val_ds   = LaneDataset(
        BASE_DIR / "clips_flat" / "val",   val_mask_dir,   augment=False)
    test_ds  = LaneDataset(
        BASE_DIR / "clips_flat" / "val",   test_mask_dir,  augment=False)

    # pin_memory only useful with CUDA; suppress warning on CPU
    pin = DEVICE.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False,
                              num_workers=1)
    return train_loader, val_loader, test_loader


# ── Step 3: Lightweight Encoder-Decoder CNN ────────────────────────────────────

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = DepthwiseSepConv(in_ch,  out_ch, stride=stride)
        self.conv2 = DepthwiseSepConv(out_ch, out_ch)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DepthwiseSepConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class LaneCNN(nn.Module):
    """
    Lightweight U-Net style encoder-decoder (~1.2 M params).
    Fast enough for ~5-10 fps on Raspberry Pi 5 CPU via ONNX.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc1      = EncoderBlock(3,   32,  downsample=False)
        self.enc2      = EncoderBlock(32,  64)
        self.enc3      = EncoderBlock(64,  128)
        self.enc4      = EncoderBlock(128, 256)
        self.bottleneck = nn.Sequential(
            DepthwiseSepConv(256, 256),
            DepthwiseSepConv(256, 256),
        )
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64)
        self.dec1 = DecoderBlock( 64,  32,  32)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.bottleneck(s4)
        d3 = self.dec3(b,  s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.head(d1)    # (B, NUM_CLASSES, H, W) raw logits


# ── Step 4: Loss & Metrics ─────────────────────────────────────────────────────

def dice_loss(logits, targets, smooth=1.0):
    probs   = F.softmax(logits, dim=1)
    B, C, H, W = probs.shape
    one_hot = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
    inter   = (probs * one_hot).sum(dim=(2, 3))
    union   = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()


def combined_loss(logits, targets):
    return F.cross_entropy(logits, targets) + dice_loss(logits, targets)


def mean_iou(logits, targets):
    preds = logits.argmax(dim=1)
    ious  = []
    for c in range(NUM_CLASSES):
        inter = ((preds == c) & (targets == c)).sum().item()
        union = ((preds == c) | (targets == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ── Step 5: Training Loop ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, scaler):
    model.train()
    total = 0.0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimiser.zero_grad()
        with torch.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda")):
            loss = combined_loss(model(images), masks)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = total_iou = 0.0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        logits = model(images)
        total_loss += combined_loss(logits, masks).item()
        total_iou  += mean_iou(logits, masks)
    n = len(loader)
    return total_loss / n, total_iou / n


def train(train_loader, val_loader):
    model     = LaneCNN().to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_iou = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss          = train_one_epoch(model, train_loader,
                                              optimiser, scaler)
        val_loss, val_iou   = evaluate(model, val_loader)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val mIoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(),
                       CHECKPOINT_DIR / "best_lane_cnn.pt")
            print(f"  ✓ Saved best model (mIoU={best_iou:.4f})")

    return model


# ── Step 6: Export to ONNX ─────────────────────────────────────────────────────

def export_onnx():
    model = LaneCNN().to("cpu")
    model.load_state_dict(torch.load(
        CHECKPOINT_DIR / "best_lane_cnn.pt", map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 3, IMG_H, IMG_W)
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model saved → {ONNX_PATH}")
    print(f"  Input  : (1, 3, {IMG_H}, {IMG_W})")
    print(f"  Output : (1, {NUM_CLASSES}, {IMG_H}, {IMG_W})  raw logits")
    print("  argmax → 0=background  1=lane")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_dataloaders()

    model = train(train_loader, val_loader)

    test_loss, test_iou = evaluate(model, test_loader)
    print(f"\nTest Loss: {test_loss:.4f} | Test mIoU: {test_iou:.4f}")

    export_onnx()

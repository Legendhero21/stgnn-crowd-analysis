"""
Convert .mat crowd dataset to YOLO format.
Windows-safe, deterministic split, data.yaml generation.
"""

import os
import cv2
import numpy as np
import scipy.io as sio
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml

IMAGES_DIR = "D:/stgnn_project/data/mat_dataset/images"
GT_DIR = "D:/stgnn_project/data/mat_dataset/ground_truth"
OUTPUT_DIR = "D:/stgnn_project/data/yolo_from_mat"

BOX_SIZE = 30
TRAIN_RATIO = 0.8
START_FRAME = None
END_FRAME = None
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_gt_points(mat_path):
    mat = sio.loadmat(mat_path)

    if "image_info" in mat:
        points = mat["image_info"][0, 0]["location"][0, 0]
    elif "annPoints" in mat:
        points = mat["annPoints"]
    elif "loc" in mat:
        points = mat["loc"]
    else:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise ValueError(f"Unknown .mat structure in {mat_path}, keys: {keys}")

    points = np.array(points, dtype=object)

    if points.size == 0:
        return np.zeros((0, 2), dtype=float)

    if points.ndim == 0:
        points = np.array([points.item()], dtype=object)

    points = np.squeeze(points)

    if points.dtype == object:
        points = np.vstack(points)

    points = np.asarray(points, dtype=float)

    if points.ndim == 2 and points.shape[1] == 3:
        points = points[:, :2]

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Invalid points shape {points.shape} in {mat_path}")

    return points


def points_to_boxes(points, img_w, img_h, box_size=BOX_SIZE):
    boxes = []

    for x, y in points:
        x1 = max(0.0, x - box_size / 2)
        y1 = max(0.0, y - box_size / 2)
        x2 = min(float(img_w), x + box_size / 2)
        y2 = min(float(img_h), y + box_size / 2)

        if x2 <= x1 or y2 <= y1:
            continue

        x_center = ((x1 + x2) / 2.0) / img_w
        y_center = ((y1 + y2) / 2.0) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        if 0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and width > 0.0 and height > 0.0:
            boxes.append((x_center, y_center, width, height))

    return boxes


def _get_frame_id_from_name(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else -1


def convert_dataset():
    print("=" * 70)
    print("MAT TO YOLO DATASET CONVERTER")
    print("=" * 70)

    if not os.path.exists(IMAGES_DIR):
        print(f"[ERROR] Images directory not found: {IMAGES_DIR}")
        return
    if not os.path.exists(GT_DIR):
        print(f"[ERROR] Ground truth directory not found: {GT_DIR}")
        return

    train_img_dir = os.path.join(OUTPUT_DIR, "images", "train")
    val_img_dir = os.path.join(OUTPUT_DIR, "images", "val")
    train_lbl_dir = os.path.join(OUTPUT_DIR, "labels", "train")
    val_lbl_dir = os.path.join(OUTPUT_DIR, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n[CONFIG]")
    print(f"  Images:       {IMAGES_DIR}")
    print(f"  Ground Truth: {GT_DIR}")
    print(f"  Output:       {OUTPUT_DIR}")
    print(f"  Box Size:     {BOX_SIZE} px")
    print(f"  Train/Val:    {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")
    print(f"  Seed:         {RANDOM_SEED}")

    image_files = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg"))

    if START_FRAME is not None and END_FRAME is not None:
        image_files = [
            f for f in image_files
            if START_FRAME <= _get_frame_id_from_name(f) <= END_FRAME
        ]

    print(f"\n[INFO] Found {len(image_files)} images to process")

    stats = {
        "total": 0,
        "train": 0,
        "val": 0,
        "total_annotations": 0,
        "skipped": 0,
        "errors": 0,
    }

    print("\n[PROCESSING]")
    for img_name in tqdm(image_files, desc="Converting"):
        try:
            img_path = os.path.join(IMAGES_DIR, img_name)
            gt_name = f"GT_{img_name.replace('.jpg', '.mat')}"
            gt_path = os.path.join(GT_DIR, gt_name)

            if not os.path.exists(gt_path):
                stats["skipped"] += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                stats["errors"] += 1
                continue

            h, w = img.shape[:2]

            points = load_gt_points(gt_path)
            if points.size == 0:
                stats["skipped"] += 1
                continue

            boxes = points_to_boxes(points, w, h, BOX_SIZE)
            if len(boxes) == 0:
                stats["skipped"] += 1
                continue

            frame_id = _get_frame_id_from_name(img_name)
            is_train = (frame_id % 10) < int(TRAIN_RATIO * 10)

            if is_train:
                dest_img_dir = train_img_dir
                dest_lbl_dir = train_lbl_dir
                stats["train"] += 1
            else:
                dest_img_dir = val_img_dir
                dest_lbl_dir = val_lbl_dir
                stats["val"] += 1

            shutil.copy(img_path, os.path.join(dest_img_dir, img_name))

            label_name = img_name.replace(".jpg", ".txt")
            label_path = os.path.join(dest_lbl_dir, label_name)
            with open(label_path, "w") as f:
                for x_c, y_c, w_box, h_box in boxes:
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w_box:.6f} {h_box:.6f}\n")

            stats["total"] += 1
            stats["total_annotations"] += len(boxes)

        except Exception as e:
            print(f"\n[ERROR] Failed to process {img_name}: {e}")
            stats["errors"] += 1
            continue

    data_yaml = {
        "path": OUTPUT_DIR,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
        "nc": 1,
    }

    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Processed images: {stats['total']}")
    print(f"  Train: {stats['train']}")
    print(f"  Val:   {stats['val']}")
    print(f"Total annotations: {stats['total_annotations']}")
    if stats["total"] > 0:
        print(f"Avg boxes/image: {stats['total_annotations']/stats['total']:.1f}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors:  {stats['errors']}")
    print(f"\nOutput dir: {OUTPUT_DIR}")
    print(f"data.yaml:  {yaml_path}")
    print("=" * 70)


def verify_labels(label_dir, num_samples=5):
    print("\n[VERIFICATION]")
    label_files = list(Path(label_dir).glob("*.txt"))
    if not label_files:
        print("No label files found.")
        return

    samples = np.random.choice(label_files, min(num_samples, len(label_files)), replace=False)

    for label_path in samples:
        print(f"Checking: {label_path.name}")
        with open(label_path) as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"  [ERROR] Line {line_num}: Expected 5 values, got {len(parts)}")
                    continue

                cls, x, y, w, h = map(float, parts)
                assert cls == 0, f"Invalid class: {cls}"
                assert 0.0 <= x <= 1.0, f"x_center out of range: {x}"
                assert 0.0 <= y <= 1.0, f"y_center out of range: {y}"
                assert 0.0 < w <= 1.0, f"width out of range: {w}"
                assert 0.0 < h <= 1.0, f"height out of range: {h}"
        print("  ✓ Valid")
    print("[OK] Verification passed")


if __name__ == "__main__":
    try:
        convert_dataset()
        verify_labels(os.path.join(OUTPUT_DIR, "labels", "train"))
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Conversion stopped by user")
    except Exception as exc:
        print(f"\n[FATAL ERROR] {exc}")
        import traceback
        traceback.print_exc()

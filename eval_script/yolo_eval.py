import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# =========================
# CONFIG
# =========================
SEQUENCE_DIR = r"D:\stgnn_project\data\txt_dataset\sequences"
ANNOTATION_DIR = r"D:\stgnn_project\data\processed\txt_json\train"

MODELS = {
    "yolov8": r"D:\stgnn_project\models\visdrone-yolov8s.pt",
    "yolo11": r"D:\stgnn_project\models\yolo11n_person_best.pt"
}

OUTPUT_DIR = "evaluation_txt_dataset"

CONF_THRESHOLD = 0.2
IMG_SIZE = 640
MAX_SEQUENCES = 5

DIST_THRESHOLD = 10  # pixels (tune if needed)


# =========================
# UTILS
# =========================
def get_box_centers(results):
    centers = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) != 0:
            continue
        x1, y1, x2, y2 = box.cpu().numpy()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))
    return centers


def match_points(pred_centers, gt_points):
    matched_gt = set()
    tp = 0

    for pc in pred_centers:
        best_dist = float("inf")
        best_idx = -1

        for i, gp in enumerate(gt_points):
            if i in matched_gt:
                continue

            dist = np.linalg.norm(np.array(pc) - np.array(gp))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_dist < DIST_THRESHOLD:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(pred_centers) - tp
    fn = len(gt_points) - tp

    return tp, fp, fn


# =========================
# EVALUATION
# =========================
def evaluate_model(model_name, model_path):
    print(f"\nEvaluating: {model_name}")
    model = YOLO(model_path)

    errors = []

    tp_total, fp_total, fn_total = 0, 0, 0

    sequences = sorted(os.listdir(SEQUENCE_DIR))[:MAX_SEQUENCES]

    for seq in sequences:
        print(f"\nProcessing sequence: {seq}")

        seq_path = os.path.join(SEQUENCE_DIR, seq)
        ann_path = os.path.join(ANNOTATION_DIR, f"{seq}.json")

        if not os.path.exists(ann_path):
            print(f"Missing annotation for {seq}, skipping")
            continue

        with open(ann_path, "r") as f:
            annotations = json.load(f)

        frame_map = {}
        for item in annotations:
            frame_map[item["frame"]] = item["points"]

        image_files = sorted(os.listdir(seq_path))

        for img_name in tqdm(image_files):
            img_path = os.path.join(seq_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            frame_id = int(os.path.splitext(img_name)[0])

            if frame_id not in frame_map:
                continue

            gt_points = frame_map[frame_id]
            gt_count = len(gt_points)

            # Inference
            results = model(img, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)[0]

            pred_centers = get_box_centers(results)
            pred_count = len(pred_centers)

            # Count metrics (REAL)
            error = pred_count - gt_count
            errors.append(error)

            # Distance matching (HEURISTIC)
            tp, fp, fn = match_points(pred_centers, gt_points)

            tp_total += tp
            fp_total += fp
            fn_total += fn

    # =========================
    # METRICS
    # =========================
    errors = np.array(errors)

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    precision = tp_total / (tp_total + fp_total + 1e-6)
    recall = tp_total / (tp_total + fn_total + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # =========================
    # SAVE
    # =========================
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "metrics.txt"), "w") as f:
        f.write("=== COUNT METRICS (Reliable) ===\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n\n")

        f.write("=== DETECTION METRICS (Heuristic) ===\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")

    print(f"\n{model_name} Results:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}

    for name, path in MODELS.items():
        results[name] = evaluate_model(name, path)

    # Comparison
    with open(os.path.join(OUTPUT_DIR, "comparison.txt"), "w") as f:
        for model, metrics in results.items():
            f.write(f"\n{model.upper()}:\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

    print("\nDone. Check evaluation_txt_dataset/")


if __name__ == "__main__":
    main()




"""import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
IMAGE_DIR = r"D:\stgnn_project\data\VisDrone2019-DET-val\images"
LABEL_DIR = r"D:\stgnn_project\data\VisDrone2019-DET-val\labels"

MODELS = {
    "yolov8": r"D:\stgnn_project\models\visdrone-yolov8s.pt",
    "yolo11": r"D:\stgnn_project\models\yolo11n_person_best.pt"
}

OUTPUT_DIR = "evaluation_output"

SAVE_SAMPLE_IMAGES = True
NUM_SAMPLE_IMAGES = 20

CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.5
IMG_SIZE = 640


# =========================
# UTILS
# =========================

def yolo_to_xyxy(box, img_w, img_h):
    x_c, y_c, w, h = box
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def draw_boxes(image, boxes, color, label):
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def load_gt(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls = int(parts[0])

        # ✅ CHANGE: merge class 0 and 1
        if cls not in [0, 1]:
            continue

        box = yolo_to_xyxy(parts[1:], img_w, img_h)
        boxes.append(box)

    return boxes


def match_boxes(pred_boxes, gt_boxes):
    matched_gt = set()
    tp = 0

    for pb in pred_boxes:
        best_iou = 0
        best_idx = -1

        for i, gb in enumerate(gt_boxes):
            if i in matched_gt:
                continue

            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= IOU_THRESHOLD:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


# =========================
# MAIN EVALUATION
# =========================

def evaluate_model(model_name, model_path):
    print(f"\nEvaluating: {model_name}")
    model = YOLO(model_path)

    tp_total, fp_total, fn_total = 0, 0, 0
    count_errors = []

    image_files = os.listdir(IMAGE_DIR)

    model_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    sample_dir = os.path.join(model_dir, "samples")
    if SAVE_SAMPLE_IMAGES:
        os.makedirs(sample_dir, exist_ok=True)

    saved_samples = 0

    for img_name in tqdm(image_files):
        img_path = os.path.join(IMAGE_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, img_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        gt_boxes = load_gt(label_path, w, h)

        results = model(img, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)[0]

        pred_boxes = []
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):

            # ✅ CHANGE: merge class 0 and 1
            if int(cls) not in [0, 1]:
                continue

            pred_boxes.append(box.cpu().numpy())

        tp, fp, fn = match_boxes(pred_boxes, gt_boxes)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        count_errors.append(len(pred_boxes) - len(gt_boxes))

        if SAVE_SAMPLE_IMAGES and saved_samples < NUM_SAMPLE_IMAGES:
            vis_img = img.copy()

            vis_img = draw_boxes(vis_img, gt_boxes, (0, 255, 0), "GT")
            vis_img = draw_boxes(vis_img, pred_boxes, (0, 0, 255), "Pred")

            save_path = os.path.join(sample_dir, f"{saved_samples}_{img_name}")
            cv2.imwrite(save_path, vis_img)

            saved_samples += 1

    precision = tp_total / (tp_total + fp_total + 1e-6)
    recall = tp_total / (tp_total + fn_total + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    mae = np.mean(np.abs(count_errors))
    rmse = np.sqrt(np.mean(np.square(count_errors)))

    with open(os.path.join(model_dir, "metrics.txt"), "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"MAE (count): {mae:.4f}\n")
        f.write(f"RMSE (count): {rmse:.4f}\n")

    cm = np.array([[tp_total, fp_total],
                   [fn_total, 0]])

    plt.imshow(cm, cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    plt.close()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae,
        "rmse": rmse
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}

    for name, path in MODELS.items():
        results[name] = evaluate_model(name, path)

    with open(os.path.join(OUTPUT_DIR, "comparison.txt"), "w") as f:
        for model, metrics in results.items():
            f.write(f"\n{model.upper()}:\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

    print("\nEvaluation complete. Check evaluation_output/ folder.")


if __name__ == "__main__":
    main()


"""
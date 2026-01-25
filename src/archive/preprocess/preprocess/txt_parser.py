import os
import json
import random

def parse_annotation_file(seq_id, annotations_dir, images_dir):
    ann_path = os.path.join(annotations_dir, f"{seq_id}.txt")
    seq_img_dir = os.path.join(images_dir, seq_id)

    if not os.path.exists(ann_path):
        print(f"[WARNING] Annotation missing for sequence {seq_id}")
        return []

    frame_data = {}
    try:
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 3:
                    continue
                frame_id, x, y = parts
                frame_id = int(frame_id)
                x, y = int(float(x)), int(float(y))

                if frame_id not in frame_data:
                    frame_data[frame_id] = {
                        "frame": frame_id,
                        "image": os.path.join(seq_img_dir, f"{frame_id:05d}.jpg"),
                        "points": []
                    }
                frame_data[frame_id]["points"].append((x, y))
    except Exception as e:
        print(f"[ERROR] Failed to parse {ann_path}: {e}")
        return []

    return list(frame_data.values())

def process_split(seq_ids, annotations_dir, sequences_dir, output_dir, split_name):
    if not seq_ids:
        print(f"[WARNING] No sequences in {split_name} split to process.")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Could not create output dir {output_dir}: {e}")
        return

    saved, skipped = 0, 0
    for seq_id in seq_ids:
        parsed_frames = parse_annotation_file(seq_id, annotations_dir, sequences_dir)
        out_path = os.path.join(output_dir, f"{seq_id}.json")
        if parsed_frames:
            try:
                with open(out_path, "w") as out_f:
                    json.dump(parsed_frames, out_f, indent=2)
                saved += 1
                print(f"[OK] Processed {seq_id} ({split_name}) → {out_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save JSON for {seq_id}: {e}")
                skipped += 1
        else:
            print(f"[SKIP] {seq_id} ({split_name}) has no annotations")
            skipped += 1

    print(f"[INFO] {split_name} split complete → {saved} saved, {skipped} skipped")

if __name__ == "__main__":
    base_dir = "D:/stgnn_project/data/txt_dataset"   # <-- adjust if needed
    annotations_dir = os.path.join(base_dir, "annotations")
    sequences_dir = os.path.join(base_dir, "sequences")
    output_dir = "D:/stgnn_project/data/processed/txt_json"

    trainlist_path = os.path.join(base_dir, "trainlist.txt")
    with open(trainlist_path, "r") as f:
        all_ids = [line.strip() for line in f.readlines()]

    if not all_ids:
        print("[ERROR] No sequences found in trainlist.txt")
        exit(1)

    random.seed(42)  # reproducible split
    random.shuffle(all_ids)
    split_idx = int(0.7 * len(all_ids))
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]

    print(f"[INFO] Total sequences: {len(all_ids)} → Train: {len(train_ids)}, Val: {len(val_ids)}")

    process_split(
        seq_ids=train_ids,
        annotations_dir=annotations_dir,
        sequences_dir=sequences_dir,
        output_dir=os.path.join(output_dir, "train"),
        split_name="train"
    )

    process_split(
        seq_ids=val_ids,
        annotations_dir=annotations_dir,
        sequences_dir=sequences_dir,
        output_dir=os.path.join(output_dir, "val"),
        split_name="val"
    )



"""

import os
import json

def parse_annotation_file(seq_id, annotations_dir, images_dir):
    
    ann_path = os.path.join(annotations_dir, f"{seq_id}.txt")
    seq_img_dir = os.path.join(images_dir, seq_id)

    if not os.path.exists(ann_path):
        print(f"[WARNING] Annotation missing for sequence {seq_id}")
        return []

    frame_data = {}
    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            frame_id, x, y = parts
            frame_id = int(frame_id)
            x, y = int(float(x)), int(float(y))

            if frame_id not in frame_data:
                frame_data[frame_id] = {
                    "frame": frame_id,
                    "image": os.path.join(seq_img_dir, f"{frame_id:05d}.jpg"),
                    "points": []
                }
            frame_data[frame_id]["points"].append((x, y))

    return list(frame_data.values())


def process_split(split_file, annotations_dir, sequences_dir, output_dir, split_name):
    
    with open(split_file, "r") as f:
        seq_ids = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    for seq_id in seq_ids:
        parsed_frames = parse_annotation_file(seq_id, annotations_dir, sequences_dir)
        if parsed_frames:
            out_path = os.path.join(output_dir, f"{seq_id}.json")
            with open(out_path, "w") as out_f:
                json.dump(parsed_frames, out_f, indent=2)
            print(f"[OK] Processed {seq_id} → {out_path}")
        else:
            print(f"[SKIP] {seq_id} has no annotations")


if __name__ == "__main__":
    base_dir = "D:/stgnn_project/data/txt_dataset"   # <-- adjust if needed
    annotations_dir = os.path.join(base_dir, "annotations")
    sequences_dir = os.path.join(base_dir, "sequences")

    output_dir = "D:/stgnn_project/data/processed/txt_json"

    # Process training split
    process_split(
        split_file=os.path.join(base_dir, "trainlist.txt"),
        annotations_dir=annotations_dir,
        sequences_dir=sequences_dir,
        output_dir=os.path.join(output_dir, "train"),
        split_name="train"
    )

    # Process test split
    process_split(
        split_file=os.path.join(base_dir, "testlist.txt"),
        annotations_dir=annotations_dir,
        sequences_dir=sequences_dir,
        output_dir=os.path.join(output_dir, "test"),
        split_name="test"
    )
"""


import os
import json
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

# === IMAGE DIMENSIONS (CRITICAL!) ===
# Adjust these to match your dataset's resolution
IMG_WIDTH = 1920   # Change if different
IMG_HEIGHT = 1080  # Change if different


def build_edges(points, method="radius", k=5, radius=0.05):  # radius in normalized coords
    """Build edges between points using kNN or radius method."""
    points = np.array(points)
    n = len(points)
    
    if n == 0 or n == 1:
        return torch.empty((2, 0), dtype=torch.long)

    try:
        if method == "knn":
            k_actual = min(k + 1, n)
            if k_actual <= 1:
                return torch.empty((2, 0), dtype=torch.long)
            
            nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm="ball_tree").fit(points)
            _, indices = nbrs.kneighbors(points)
            edges = []
            for i in range(n):
                for j in indices[i][1:]:
                    edges.append([i, j])
            edges = np.array(edges).T if edges else np.empty((2, 0), dtype=int)
            
        else:  # radius method
            nbrs = NearestNeighbors(radius=radius, algorithm="ball_tree").fit(points)
            _, indices = nbrs.radius_neighbors(points)
            edges = []
            for i, neighbors in enumerate(indices):
                for j in neighbors:
                    if i != j:
                        edges.append([i, j])
            edges = np.array(edges).T if edges else np.empty((2, 0), dtype=int)

        return torch.tensor(edges, dtype=torch.long)
    
    except Exception as e:
        print(f"[WARNING] Edge building failed: {e}")
        return torch.empty((2, 0), dtype=torch.long)


def normalize_coordinates(points, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    """Normalize pixel coordinates to [0, 1] range."""
    normalized = []
    for x, y in points:
        norm_x = x / img_width
        norm_y = y / img_height
        normalized.append((norm_x, norm_y))
    return normalized


def compute_density(points, radius=0.05):  # radius in normalized space
    """Density = number of neighbors within radius for each node."""
    if not points:
        return []
    
    densities = []
    for i, (x, y) in enumerate(points):
        count = 0
        for j, (xx, yy) in enumerate(points):
            if i != j and (x - xx) ** 2 + (y - yy) ** 2 <= radius ** 2:
                count += 1
        densities.append(float(count) / 10.0)  # Normalize density to ~[0, 1]
    
    return densities


def subsample_to_fixed_size(points, target_size):
    """Subsample or upsample points to exactly target_size."""
    n = len(points)
    
    if n == target_size:
        return list(range(n))
    elif n > target_size:
        return np.random.choice(n, target_size, replace=False).tolist()
    else:
        base = list(range(n))
        extras = np.random.choice(n, target_size - n, replace=True).tolist()
        return base + extras


def json_to_graph_normalized(json_path, method="radius", k=5, radius=0.05, 
                             img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    """
    Convert JSON to graphs with NORMALIZED coordinates and velocities.
    All values will be in [0, 1] range for stable training.
    """
    try:
        with open(json_path, "r") as f:
            frames = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read {json_path}: {e}")
        return []

    if not frames or not isinstance(frames, list):
        return []

    # Step 1: Determine target node count
    all_points = []
    valid_frames = []
    
    for frame in frames:
        if isinstance(frame, dict) and "points" in frame:
            points = frame.get("points", [])
            if points and len(points) >= 10:
                all_points.append(len(points))
                valid_frames.append(frame)
    
    if not valid_frames or not all_points:
        return []
    
    target_nodes = int(np.median(all_points))
    if target_nodes < 20:
        return []
    
    print(f"[INFO] Processing with fixed {target_nodes} nodes per frame (normalized coords)")
    
    # Step 2: Build graphs with NORMALIZED coordinates
    graphs = []
    prev_points_norm = None
    
    for frame_idx, frame in enumerate(valid_frames):
        points = frame.get("points", [])
        
        try:
            # Normalize coordinates to [0, 1]
            points_norm = normalize_coordinates(points, img_width, img_height)
            
            # Subsample to fixed size
            indices = subsample_to_fixed_size(points_norm, target_nodes)
            sampled_points = [points_norm[i] for i in indices]
            
            # Compute velocities (in normalized space)
            if prev_points_norm is not None:
                velocities = []
                for i, (x, y) in enumerate(sampled_points):
                    if i < len(prev_points_norm):
                        px, py = prev_points_norm[i]
                        dx = x - px
                        dy = y - py
                        velocities.append((dx, dy))
                    else:
                        velocities.append((0.0, 0.0))
            else:
                velocities = [(0.0, 0.0)] * target_nodes
            
            # Compute density (in normalized space)
            densities = compute_density(sampled_points, radius=radius)
            
            # Build features [x_norm, y_norm, dx_norm, dy_norm, density_norm]
            # All values now in approximately [0, 1] or [-0.1, 0.1] for velocities
            features = []
            for (x, y), (dx, dy), d in zip(sampled_points, velocities, densities):
                features.append([float(x), float(y), float(dx), float(dy), float(d)])
            
            x_tensor = torch.tensor(features, dtype=torch.float)
            edge_index = build_edges(sampled_points, method=method, k=k, radius=radius)
            
            data = Data(x=x_tensor, edge_index=edge_index)
            data.frame_id = frame.get("frame", frame_idx)
            data.image_path = frame.get("image", "")
            
            graphs.append(data)
            prev_points_norm = sampled_points
            
        except Exception as e:
            print(f"[WARNING] Failed frame {frame_idx}: {e}")
            continue
    
    return graphs


def process_all_json(input_dir, output_dir, method="radius", k=5, radius=0.05,
                     img_width=IMG_WIDTH, img_height=IMG_HEIGHT):
    """Convert all JSONs → PyG graphs with NORMALIZED features."""
    if not os.path.exists(input_dir):
        print(f"[WARNING] Input dir missing: {input_dir}")
        return 0, 0

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    if not json_files:
        print(f"[WARNING] No JSON files in {input_dir}")
        return 0, 0

    os.makedirs(output_dir, exist_ok=True)
    saved, skipped = 0, 0

    for file in json_files:
        seq_id = os.path.splitext(file)[0]
        json_path = os.path.join(input_dir, file)

        try:
            graphs = json_to_graph_normalized(
                json_path, 
                method=method, 
                k=k, 
                radius=radius,
                img_width=img_width,
                img_height=img_height
            )
            
            if graphs and len(graphs) >= 2:
                out_path = os.path.join(output_dir, f"{seq_id}.pt")
                torch.save(graphs, out_path)
                saved += 1
                print(f"[OK] {seq_id} → {len(graphs)} frames (normalized)")
            else:
                skipped += 1
                
        except Exception as e:
            skipped += 1
            print(f"[ERROR] {seq_id}: {e}")

    return saved, skipped


if __name__ == "__main__":
    base_dir = "D:/stgnn_project/data/processed/txt_json"
    output_dir = "D:/stgnn_project/data/processed/txt_graphs"

    # === IMPORTANT: Set your image dimensions here! ===
    print(f"\n[CONFIG] Image dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
    print("[INFO] If this is wrong, update IMG_WIDTH and IMG_HEIGHT at top of file!\n")

    splits = ["train", "val"]

    for split in splits:
        print(f"{'='*60}")
        print(f"[INFO] Processing {split} split...")
        print(f"{'='*60}")
        
        in_dir = os.path.join(base_dir, split)
        out_dir = os.path.join(output_dir, split)
        
        saved, skipped = process_all_json(
            in_dir, out_dir, 
            method="radius", 
            radius=0.05,  # In normalized space (~50px at 1920x1080)
            k=5
        )
        
        print(f"\n[SUMMARY] {split.upper()} → {saved} saved, {skipped} skipped\n")

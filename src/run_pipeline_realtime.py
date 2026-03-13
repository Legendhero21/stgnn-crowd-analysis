"""
Real-Time Crowd Anomaly Detection Pipeline
===========================================
YOLOv11n + ByteTrack + kNN Graph + Padded Temporal Buffer + STGNN (ONNX)

Improvements over the original pipeline:
1. ByteTrack tracking for stable person IDs  (Zhang et al., 2022)
2. kNN graph instead of fixed-radius graph   (Mohamed et al., 2020)
3. Padded temporal buffer — NEVER resets on node-count change
                                              (Wu et al., 2020)
4. Expanded 8-feature node vectors           (Helbing & Molnár, 1995)
5. Temporal edge support (disabled by default) (Yan et al., 2018)
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OMP runtime conflict (Issue 4)

import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist

# Add models directory to path for STGNN import
_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

from alert_logic import StampedeAlert
from crowd_metrics import CrowdMetrics
from temporal_buffer import TemporalGraphBuffer, MAX_NODES

try:
    from bytetrack_tracker import ByteTrackTracker, TrackedPerson
except ImportError as exc:
    print(f"[ERROR] Cannot import bytetrack_tracker.py: {exc}")
    sys.exit(1)

# Issue 6 — warn if 'lap' package is missing (ByteTrack dependency)
try:
    import lap  # noqa: F401
except ImportError:
    print(
        "[WARNING] ByteTrack dependency 'lap' not found. "
        "Install it with: pip install lap>=0.5.12\n"
        "Restart environment if tracking fails."
    )


# ==========================================================
# LOGGING
# ==========================================================
logger = logging.getLogger("PIPELINE")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - PIPELINE - %(levelname)s - %(message)s",
    )


# ==========================================================
# CONFIGURATION
# ==========================================================

YOLO_MODEL = os.getenv(
    "YOLO_MODEL_PATH",
    "D:/stgnn_project/models/yolo11n_person_best.pt",
)
STGNN_MODEL = os.getenv(
    "STGNN_MODEL_PATH",
    "D:/stgnn_project/src/archive/stgnn_training/checkpoints/stgnn_best.pth",
)
VIDEO_SOURCE = os.getenv(
    "VIDEO_SOURCE",
    "D:/stgnn_project/data/videos/test_crowd.mp4",
)

OUTPUT_DIR = os.getenv(
    "PIPELINE_OUTPUT_DIR",
    "D:/stgnn_project/outputs/pipeline_results_shibuya",
)

SAVE_OUTPUT_VIDEO = True
OUTPUT_FPS = 10

# --- Graph ---
GRAPH_K = 5                       # kNN neighbor count
MIN_NODES = 2                     # minimum detections to build a graph
MAX_VELOCITY = 0.1                # clip velocity in normalized coords
ENABLE_TEMPORAL_EDGES = False     # disabled — current ONNX not trained with them

# --- Display ---
ANOMALY_THRESHOLD_WARNING = 0.5
ANOMALY_THRESHOLD_CRITICAL = 2.0
DISPLAY_GRAPH_EDGES = True
DISPLAY_FPS = True
FRAME_DELAY = 1  # ms for cv2.waitKey
TEMPORAL_WINDOW = 5

# --- Debug ---
DEBUG_MODE = True  # Enable detailed per-frame console + visual debug

# --- Legacy model compat ---
LEGACY_FEATURE_COUNT = 5          # stgnn_best.pth expects 5 features

# --- STGNN architecture (must match checkpoint) ---
STGNN_CONFIG = {
    "in_channels": 5,
    "hidden_channels": 64,
    "out_channels": 2,
    "num_layers": 3,
    "dropout": 0.1,
    "kernel_size": 3,
}


# ==========================================================
# VALIDATION
# ==========================================================

def validate_setup() -> None:
    logger.info("=" * 70)
    logger.info("REAL-TIME CROWD ANOMALY DETECTION PIPELINE (v2)")
    logger.info("=" * 70)

    missing = False
    if not os.path.isfile(YOLO_MODEL):
        logger.error("YOLOv11 model not found: %s", YOLO_MODEL)
        missing = True

    if not os.path.isfile(STGNN_MODEL):
        logger.error("STGNN model not found: %s", STGNN_MODEL)
        missing = True

    if not os.path.isfile(VIDEO_SOURCE):
        logger.error("Video source not found: %s", VIDEO_SOURCE)
        missing = True

    if missing:
        sys.exit(1)

    logger.info("[OK] YOLOv11 model: %s", YOLO_MODEL)
    logger.info("[OK] STGNN model: %s", STGNN_MODEL)
    logger.info("[OK] Video source: %s", VIDEO_SOURCE)
    logger.info("=" * 70 + "\n")


# ==========================================================
# GRAPH BUILDER  (kNN + expanded features)
# ==========================================================

class RealtimeGraphBuilder:
    """
    Build a padded kNN graph from tracked persons.

    Node features (8-dim):
        [x, y, dx, dy, speed, heading, local_density, bbox_area]

    Edge construction:
        - kNN with k_effective = min(GRAPH_K, N-1)
        - Edges only between real (mask==1) nodes
        - Padded nodes are completely excluded
    """

    def __init__(self, k: int = GRAPH_K, max_nodes: int = MAX_NODES) -> None:
        self.k = int(k)
        self.max_nodes = int(max_nodes)

    # ------------------------------------------------------------------
    def build_graph(
        self,
        tracked_persons: List[TrackedPerson],
        frame_shape: Tuple[int, int],
        prev_positions: Dict[int, Tuple[float, float]],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Build padded kNN graph from tracked persons.

        Args:
            tracked_persons: output from ByteTrackTracker.update().
            frame_shape: (H, W).
            prev_positions: {track_id: (cx_norm, cy_norm)} from previous frame.

        Returns:
            Dict with 'x' [MAX_NODES, 8], 'mask' [MAX_NODES],
            'edge_index' [2, E], 'track_ids' [N_valid].
            Returns None if fewer than MIN_NODES detected.
        """
        if not tracked_persons or len(tracked_persons) < MIN_NODES:
            return None

        h, w = frame_shape
        if h <= 0 or w <= 0:
            return None

        n_actual = min(len(tracked_persons), self.max_nodes)
        persons = tracked_persons[:n_actual]

        # --- Normalize coordinates ---
        coords = np.zeros((n_actual, 2), dtype=np.float32)
        bbox_areas = np.zeros(n_actual, dtype=np.float32)
        track_ids = np.zeros(n_actual, dtype=np.int64)

        for i, p in enumerate(persons):
            coords[i, 0] = p.cx / float(w)
            coords[i, 1] = p.cy / float(h)
            # Normalized bbox area
            bw = (p.x2 - p.x1) / float(w)
            bh = (p.y2 - p.y1) / float(h)
            bbox_areas[i] = bw * bh
            track_ids[i] = p.track_id

        # --- Compute velocity from tracked IDs ---
        velocity = np.zeros((n_actual, 2), dtype=np.float32)
        for i, p in enumerate(persons):
            tid = p.track_id
            if tid in prev_positions:
                prev_x, prev_y = prev_positions[tid]
                dx = coords[i, 0] - prev_x
                dy = coords[i, 1] - prev_y
                # Clip large jumps (tracking loss)
                dx = float(np.clip(dx, -MAX_VELOCITY, MAX_VELOCITY))
                dy = float(np.clip(dy, -MAX_VELOCITY, MAX_VELOCITY))
                velocity[i, 0] = dx
                velocity[i, 1] = dy

        # --- Derived features ---
        speed = np.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2)
        heading = np.arctan2(velocity[:, 1], velocity[:, 0]) / np.pi  # [-1, 1]

        # --- Build kNN edges (real nodes only) ---
        edge_index = self._build_knn_edges(coords)

        # --- Local density = neighbors / k ---
        local_density = self._compute_density(n_actual, edge_index)

        # --- Assemble 8-feature vector ---
        features = np.hstack([
            coords,                 # [N, 2]  x, y
            velocity,               # [N, 2]  dx, dy
            speed[:, None],         # [N, 1]
            heading[:, None],       # [N, 1]
            local_density[:, None], # [N, 1]
            bbox_areas[:, None],    # [N, 1]
        ]).astype(np.float32)       # [N, 8]

        # --- Pad to MAX_NODES ---
        n_feat = features.shape[1]
        x_padded = np.zeros((self.max_nodes, n_feat), dtype=np.float32)
        mask = np.zeros(self.max_nodes, dtype=np.float32)

        x_padded[:n_actual] = features
        mask[:n_actual] = 1.0

        result = {
            "x": x_padded,          # [MAX_NODES, 8]
            "mask": mask,            # [MAX_NODES]
            "edge_index": edge_index,  # [2, E]
            "track_ids": track_ids,    # [N_valid]
            "_debug_features": features,  # [N_actual, 8] unpadded (debug only)
        }

        if DEBUG_MODE:
            real_nodes = int(mask.sum())
            padded_nodes = self.max_nodes - real_nodes
            print(
                f"[GRAPH] real_nodes: {real_nodes} | padded: {padded_nodes} "
                f"| edges: {edge_index.shape[1]} "
                f"| density mean: {local_density.mean():.3f} "
                f"| speed mean: {speed.mean():.4f}"
            )

        return result

    # ------------------------------------------------------------------
    def _build_knn_edges(self, coords: np.ndarray) -> np.ndarray:
        """
        Build symmetric kNN edges.  k_effective = min(k, N-1).
        Only considers real (non-padded) nodes.
        """
        n = coords.shape[0]
        if n < 2:
            return np.zeros((2, 0), dtype=np.int64)

        k_eff = min(self.k, n - 1)

        dists = cdist(coords, coords)
        # Set self-distance to inf so a node isn't its own neighbor
        np.fill_diagonal(dists, np.inf)

        rows, cols = [], []
        for i in range(n):
            # k nearest neighbors for node i
            neighbors = np.argpartition(dists[i], k_eff)[:k_eff]
            for j in neighbors:
                rows.append(i)
                cols.append(j)

        if not rows:
            return np.zeros((2, 0), dtype=np.int64)

        # Symmetrize (add reverse edges)
        rows_sym = rows + cols
        cols_sym = cols + rows
        edge_index = np.stack([rows_sym, cols_sym], axis=0).astype(np.int64)

        # Remove duplicates
        edge_set = set()
        unique_rows, unique_cols = [], []
        for r, c in zip(edge_index[0], edge_index[1]):
            if (r, c) not in edge_set:
                edge_set.add((r, c))
                unique_rows.append(r)
                unique_cols.append(c)

        if not unique_rows:
            return np.zeros((2, 0), dtype=np.int64)

        return np.stack([unique_rows, unique_cols], axis=0).astype(np.int64)

    # ------------------------------------------------------------------
    def _compute_density(self, n: int, edge_index: np.ndarray) -> np.ndarray:
        """
        Local density = unique_neighbors / k  (scale-invariant, in [0,1]).

        After symmetrization, each undirected edge (i,j) appears as both
        (i→j) and (j→i).  We count unique neighbors per node to avoid
        double-counting, then divide by k.
        """
        density = np.zeros(n, dtype=np.float32)
        if edge_index.shape[1] > 0:
            # Count unique neighbors per node (not raw degree)
            for i in range(n):
                neighbors = edge_index[1, edge_index[0] == i]
                density[i] = len(np.unique(neighbors))
        if self.k > 0:
            density /= float(self.k)
        # Enforce [0, 1] bounds
        density = np.clip(density, 0.0, 1.0)
        return density


# ==========================================================
# STGNN INFERENCE
# ==========================================================

class STGNNInference:
    """PyTorch-based STGNN inference (replaces ONNX).

    Loads a .pth checkpoint and runs inference with torch.no_grad().
    Slices to LEGACY_FEATURE_COUNT features for backward compatibility
    with models trained on 5-feature input.
    """

    def __init__(self, model_path: str) -> None:
        from stgnn import STGNN  # models/stgnn.py

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model with the same config used during training
        self.model = STGNN(**STGNN_CONFIG)

        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            # Handle both raw state_dict and wrapped checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info("Loaded STGNN checkpoint from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load STGNN checkpoint from %s: %s", model_path, exc)
            raise

        self.model.to(self.device)
        self.model.eval()

        self._model_features = STGNN_CONFIG["in_channels"]
        logger.info(
            "STGNN ready  device=%s  in_channels=%d  T=%d",
            self.device,
            self._model_features,
            TEMPORAL_WINDOW,
        )

        if self._model_features == LEGACY_FEATURE_COUNT:
            logger.warning(
                "Using legacy ST-GNN model (%d features). "
                "Extra features will be ignored.",
                self._model_features,
            )

    def predict_from_sequence(
        self,
        x_seq: np.ndarray,
        edge_index: np.ndarray,
    ) -> float:
        """
        Run STGNN inference on a temporal sequence.

        Args:
            x_seq: (1, T, N, F) numpy array
            edge_index: (2, E) numpy array

        Returns:
            anomaly_score: float (MSE × 1000)
        """
        if x_seq is None or x_seq.size == 0:
            return 0.0
        if edge_index is None:
            return 0.0

        if x_seq.ndim != 4 or x_seq.shape[0] != 1:
            logger.warning("Unexpected x_seq shape: %s", x_seq.shape)
            return 0.0
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            logger.warning("Unexpected edge_index shape: %s", edge_index.shape)
            return 0.0

        # --- Legacy model compatibility: slice to 5 features ---
        actual_features = x_seq.shape[3]
        if actual_features > self._model_features:
            x_seq = x_seq[:, :, :, :self._model_features]

        # Convert to torch tensors
        x_tensor = torch.from_numpy(x_seq).float().to(self.device)
        edge_tensor = torch.from_numpy(edge_index).long().to(self.device)

        try:
            with torch.no_grad():
                preds = self.model(x_tensor, edge_tensor)  # [1, N, 2]
        except Exception as exc:
            logger.error("STGNN inference failed: %s", exc)
            return 0.0

        preds = preds.cpu().numpy()  # [1, N, 2]

        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]  # [N, 2]

        # Compare predicted (x, y) with last frame's actual positions
        n_feat_used = min(self._model_features, 2)
        last_x = x_seq[0, -1, :, :n_feat_used]  # [N, 2]

        if preds.shape[0] != last_x.shape[0]:
            min_nodes = min(preds.shape[0], last_x.shape[0])
            preds = preds[:min_nodes]
            last_x = last_x[:min_nodes]

        mse = float(np.mean((preds[:, :n_feat_used] - last_x) ** 2))
        scaled_mse = mse * 1000.0

        return scaled_mse


# ==========================================================
# VISUALIZATION
# ==========================================================

class CrowdVisualizer:
    COLORS = {
        "normal": (0, 255, 0),
        "warning": (0, 255, 255),
        "critical": (0, 0, 255),
    }

    def draw(
        self,
        frame: np.ndarray,
        tracked_persons: List[TrackedPerson],
        graph: Optional[Dict[str, np.ndarray]],
        anomaly: float,
        state: str,
    ) -> np.ndarray:
        vis = frame.copy()

        if anomaly > ANOMALY_THRESHOLD_CRITICAL:
            alert = "critical"
        elif anomaly > ANOMALY_THRESHOLD_WARNING:
            alert = "warning"
        else:
            alert = "normal"

        color = self.COLORS[alert]

        # Draw graph edges
        if DISPLAY_GRAPH_EDGES and graph is not None and len(tracked_persons) > 1:
            edge_index = graph.get("edge_index")
            if edge_index is not None and edge_index.size > 0:
                for s, d in edge_index.T:
                    if 0 <= s < len(tracked_persons) and 0 <= d < len(tracked_persons):
                        p1 = (int(tracked_persons[s].cx), int(tracked_persons[s].cy))
                        p2 = (int(tracked_persons[d].cx), int(tracked_persons[d].cy))
                        cv2.line(vis, p1, p2, (80, 80, 80), 1)

        # Draw persons: bounding box + center + track ID + velocity arrow
        h_vis, w_vis = vis.shape[:2]
        debug_feats = graph.get("_debug_features") if graph is not None else None

        for idx, p in enumerate(tracked_persons):
            cx, cy = int(p.cx), int(p.cy)

            # Bounding box
            cv2.rectangle(
                vis,
                (int(p.x1), int(p.y1)),
                (int(p.x2), int(p.y2)),
                color,
                1,
            )

            # Center dot
            cv2.circle(vis, (cx, cy), 4, color, -1)

            # Track ID label
            cv2.putText(
                vis,
                f"ID:{p.track_id}",
                (cx + 7, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            # Velocity arrow (if features available)
            if debug_feats is not None and idx < debug_feats.shape[0]:
                dx_norm = debug_feats[idx, 2]  # dx in normalized coords
                dy_norm = debug_feats[idx, 3]  # dy in normalized coords
                # Scale to pixel space and amplify for visibility
                arrow_scale = 800.0
                end_x = cx + int(dx_norm * arrow_scale)
                end_y = cy + int(dy_norm * arrow_scale)
                cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), (0, 200, 255), 2, tipLength=0.3)

        n_real = len(tracked_persons)
        cv2.putText(
            vis,
            f"People: {n_real} | Anomaly: {anomaly:.5f} | {alert.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        cv2.putText(
            vis,
            f"STATE: {state}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if state == "STAMPEDE"
            else (0, 255, 255) if state == "UNSTABLE"
            else (0, 255, 0),
            2,
        )

        return vis


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    validate_setup()

    # --- Initialize components ---
    tracker = ByteTrackTracker(
        model_path=YOLO_MODEL,
        conf_threshold=0.4,
        device="cuda",
    )

    graph_builder = RealtimeGraphBuilder(k=GRAPH_K, max_nodes=MAX_NODES)
    stgnn = STGNNInference(STGNN_MODEL)
    visualizer = CrowdVisualizer()
    temporal_buffer = TemporalGraphBuffer(
        window_size=TEMPORAL_WINDOW,
        max_nodes=MAX_NODES,
    )
    alert_engine = StampedeAlert()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error("Failed to open video: %s", VIDEO_SOURCE)
        return

    writer = None
    if SAVE_OUTPUT_VIDEO:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            logger.warning("Invalid video dimensions; disabling video writing.")
        else:
            out_path = os.path.join(OUTPUT_DIR, "pipeline_output.mp4")
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                OUTPUT_FPS,
                (w, h),
            )
            logger.info("Saving output video to: %s", out_path)

    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video or read error.")
                break

            frame_idx += 1

            # ----- 1. DETECT + TRACK -----
            tracked_persons = tracker.update(frame)

            # ----- 2. BUILD GRAPH -----
            # Get previous positions for velocity
            prev_positions = tracker.get_previous_positions()

            graph = graph_builder.build_graph(
                tracked_persons,
                frame.shape[:2],
                prev_positions,
            )

            # Store current positions for next frame
            # Merge with existing positions so IDs from temporarily
            # lost tracks are preserved (Issue 3)
            if tracked_persons:
                h_frame, w_frame = frame.shape[:2]
                current_positions = tracker.get_previous_positions()  # keep old
                current_positions.update({                            # overwrite with new
                    p.track_id: (p.cx / float(w_frame), p.cy / float(h_frame))
                    for p in tracked_persons
                })
                tracker.store_positions(current_positions)
            # When no persons detected, keep previous positions intact
            # (do NOT wipe — they'll be used when people reappear)

            # ----- 3. TEMPORAL BUFFER -----
            # If no persons detected, graph is None → push returns (None, None)
            x_seq, mask_seq = temporal_buffer.push(graph)

            if DEBUG_MODE:
                buf_len = len(temporal_buffer.buffer)
                if x_seq is None:
                    if graph is None:
                        print(f"[BUFFER] Frame {frame_idx}: no detections → skipped")
                    else:
                        print(f"[BUFFER] Frame {frame_idx}: filling {buf_len}/{TEMPORAL_WINDOW}")
                else:
                    print(
                        f"[BUFFER] Frame {frame_idx}: READY "
                        f"x_seq={x_seq.shape} mask_seq={mask_seq.shape}"
                    )

            # ----- 4. STGNN INFERENCE -----
            if x_seq is not None and graph is not None:
                if DEBUG_MODE:
                    print(f"[STGNN] Running inference  x_seq={x_seq.shape}  "
                          f"edges={graph['edge_index'].shape}")

                anomaly = stgnn.predict_from_sequence(
                    x_seq,
                    graph["edge_index"],
                )

                if DEBUG_MODE:
                    print(f"[STGNN] Anomaly score: {anomaly:.5f}")
            else:
                anomaly = 0.0

            # ----- 5. METRICS & ALERT -----
            metrics = CrowdMetrics.compute(graph)
            state = alert_engine.update(anomaly, metrics)

            # ----- 6. VISUALIZE -----
            vis = visualizer.draw(frame, tracked_persons, graph, anomaly, state)

            if DISPLAY_FPS:
                dt = time.time() - t0
                fps = frame_idx / dt if dt > 0 else 0.0
                cv2.putText(
                    vis,
                    f"FPS: {fps:.1f}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Crowd Anomaly Detection", vis)

            if writer is not None:
                writer.write(vis)

            key = cv2.waitKey(FRAME_DELAY) & 0xFF
            if key == ord("q"):
                logger.info("User requested exit.")
                break

            if frame_idx % 200 == 0:
                logger.info(
                    "Frame %d | anomaly=%.5f | state=%s | people=%d",
                    frame_idx,
                    anomaly,
                    state,
                    len(tracked_persons),
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Frames processed: %d", frame_idx)
        logger.info("[SUCCESS] Pipeline finished.")


if __name__ == "__main__":
    main()

"""
Graph Builder
-------------
Builds spatial graphs from person detections for STGNN.

This is the canonical implementation used by EdgeClient.
Logic mirrors the original RealtimeGraphBuilder from run_pipeline_realtime.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.spatial.distance import cdist


logger = logging.getLogger(__name__)

# Constants
MAX_NODES: int = 100
MAX_VELOCITY = 0.1

class GraphBuilder:
    """
    Build a padded kNN graph from tracked persons.
    Matches RealtimeGraphBuilder implementation.
    """
    
    def __init__(self, radius: float = 0.05, min_nodes: int = 2, max_nodes: int = MAX_NODES) -> None:
        self.k = 5 # default neighbor count
        self.min_nodes = min_nodes
        self.max_nodes = int(max_nodes)
    
    def build_graph(
        self,
        tracked_persons: List[Any],
        frame_shape: Tuple[int, int],
        prev_positions: Dict[int, Tuple[float, float]] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        if not tracked_persons or len(tracked_persons) < self.min_nodes:
            return None
            
        if prev_positions is None:
            prev_positions = {}
        
        h, w = frame_shape
        if h <= 0 or w <= 0:
            return None
        
        n_actual = min(len(tracked_persons), self.max_nodes)
        persons = tracked_persons[:n_actual]
        
        coords = np.zeros((n_actual, 2), dtype=np.float32)
        bbox_areas = np.zeros(n_actual, dtype=np.float32)
        track_ids = np.zeros(n_actual, dtype=np.int64)
        
        for i, p in enumerate(persons):
            coords[i, 0] = p.cx / float(w)
            coords[i, 1] = p.cy / float(h)
            bw = (p.x2 - p.x1) / float(w)
            bh = (p.y2 - p.y1) / float(h)
            bbox_areas[i] = bw * bh
            track_ids[i] = p.track_id
            
        velocity = np.zeros((n_actual, 2), dtype=np.float32)
        for i, p in enumerate(persons):
            tid = p.track_id
            if tid in prev_positions:
                prev_x, prev_y = prev_positions[tid]
                dx = coords[i, 0] - prev_x
                dy = coords[i, 1] - prev_y
                dx = float(np.clip(dx, -MAX_VELOCITY, MAX_VELOCITY))
                dy = float(np.clip(dy, -MAX_VELOCITY, MAX_VELOCITY))
                velocity[i, 0] = dx
                velocity[i, 1] = dy
                
        speed = np.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2)
        heading = np.arctan2(velocity[:, 1], velocity[:, 0]) / np.pi
        
        edge_index = self._build_knn_edges(coords)
        local_density = self._compute_density(n_actual, edge_index)
        
        features = np.hstack([
            coords,                 # [N, 2]
            velocity,               # [N, 2]
            speed[:, None],         # [N, 1]
            heading[:, None],       # [N, 1]
            local_density[:, None], # [N, 1]
            bbox_areas[:, None],    # [N, 1]
        ]).astype(np.float32)
        
        n_feat = features.shape[1]
        x_padded = np.zeros((self.max_nodes, n_feat), dtype=np.float32)
        mask = np.zeros(self.max_nodes, dtype=np.float32)
        
        x_padded[:n_actual] = features
        mask[:n_actual] = 1.0
        
        return {
            "x": x_padded,
            "mask": mask,
            "edge_index": edge_index,
            "track_ids": track_ids,
        }
        
    def _build_knn_edges(self, coords: np.ndarray) -> np.ndarray:
        n = coords.shape[0]
        if n < 2:
            return np.zeros((2, 0), dtype=np.int64)
            
        k_eff = min(self.k, n - 1)
        dists = cdist(coords, coords)
        np.fill_diagonal(dists, np.inf)
        
        rows, cols = [], []
        for i in range(n):
            neighbors = np.argpartition(dists[i], k_eff)[:k_eff]
            for j in neighbors:
                rows.append(i)
                cols.append(j)
                
        if not rows:
            return np.zeros((2, 0), dtype=np.int64)
            
        rows_sym = rows + cols
        cols_sym = cols + rows
        edge_index = np.stack([rows_sym, cols_sym], axis=0).astype(np.int64)
        
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
        
    def _compute_density(self, n: int, edge_index: np.ndarray) -> np.ndarray:
        density = np.zeros(n, dtype=np.float32)
        if edge_index.shape[1] > 0:
            for i in range(n):
                neighbors = edge_index[1, edge_index[0] == i]
                density[i] = len(np.unique(neighbors))
        if self.k > 0:
            density /= float(self.k)
        density = np.clip(density, 0.0, 1.0)
        return density
        
    def reset(self) -> None:
        pass
    
    @property
    def current_node_count(self) -> int:
        return 0

"""
Unit Tests for Pipeline Components
-----------------------------------
Tests for the improved pipeline:
1. TemporalGraphBuffer (padding, masking, no reset on node count change)
2. RealtimeGraphBuilder (kNN, expanded features, masked edges)
3. CrowdMetrics (mask-aware, auto feature detection)

Run: python -m pytest tests/test_pipeline_components.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src is importable
_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from temporal_buffer import TemporalGraphBuffer, MAX_NODES
from crowd_metrics import CrowdMetrics, FeatureIndex


# ===========================================================
# Helpers
# ===========================================================

def _make_graph(n_nodes: int, n_feat: int = 8) -> dict:
    """Create a synthetic graph dict with n_nodes real nodes."""
    return {"x": np.random.randn(n_nodes, n_feat).astype(np.float32)}


# ===========================================================
# TemporalGraphBuffer Tests
# ===========================================================

class TestTemporalGraphBuffer:
    """Verify padding/masking temporal buffer behaviour."""

    def test_buffer_does_not_reset_on_node_count_change(self):
        """Core fix: changing node count should NOT clear the buffer."""
        buf = TemporalGraphBuffer(window_size=3, max_nodes=20)

        buf.push(_make_graph(5))   # 5 nodes
        assert len(buf.buffer) == 1

        buf.push(_make_graph(7))   # 7 nodes — different!
        assert len(buf.buffer) == 2, "Buffer must NOT reset when node count changes"

        x_seq, mask_seq = buf.push(_make_graph(3))  # 3 nodes — different again
        assert len(buf.buffer) == 3
        assert x_seq is not None, "Buffer should be full after 3 pushes"

    def test_output_shape(self):
        """x_seq and mask_seq must have the correct padded shape."""
        max_n = 30
        buf = TemporalGraphBuffer(window_size=2, max_nodes=max_n)

        buf.push(_make_graph(10, 8))
        x_seq, mask_seq = buf.push(_make_graph(15, 8))

        assert x_seq is not None
        assert x_seq.shape == (1, 2, max_n, 8)
        assert mask_seq.shape == (1, 2, max_n)

    def test_mask_correctness(self):
        """mask must be 1 for real nodes and 0 for padded nodes."""
        buf = TemporalGraphBuffer(window_size=1, max_nodes=10)

        x_seq, mask_seq = buf.push(_make_graph(4, 5))
        assert x_seq is not None
        # First 4 should be 1, rest should be 0
        np.testing.assert_array_equal(mask_seq[0, 0, :4], 1.0)
        np.testing.assert_array_equal(mask_seq[0, 0, 4:], 0.0)

    def test_padded_features_are_zero(self):
        """Padded node features must be exactly zero."""
        buf = TemporalGraphBuffer(window_size=1, max_nodes=10)
        x_seq, _ = buf.push(_make_graph(3, 5))
        np.testing.assert_array_equal(x_seq[0, 0, 3:, :], 0.0)

    def test_none_graph_skipped(self):
        """Pushing None should not add to buffer (empty frame skip)."""
        buf = TemporalGraphBuffer(window_size=3, max_nodes=10)
        buf.push(_make_graph(5))
        x, m = buf.push(None)
        assert x is None
        assert len(buf.buffer) == 1, "None push should not add to buffer"

    def test_empty_graph_skipped(self):
        """Graph with 0 nodes should be skipped."""
        buf = TemporalGraphBuffer(window_size=3, max_nodes=10)
        buf.push(_make_graph(5))
        x, m = buf.push({"x": np.zeros((0, 5), dtype=np.float32)})
        assert x is None
        assert len(buf.buffer) == 1

    def test_truncation_when_exceeding_max_nodes(self):
        """If real nodes > MAX_NODES, truncate to MAX_NODES."""
        max_n = 5
        buf = TemporalGraphBuffer(window_size=1, max_nodes=max_n)
        x_seq, mask_seq = buf.push(_make_graph(10, 4))
        assert x_seq.shape == (1, 1, max_n, 4)
        np.testing.assert_array_equal(mask_seq[0, 0, :], 1.0)  # all 5 are valid

    def test_window_sliding(self):
        """Buffer should discard oldest frame once window is exceeded."""
        buf = TemporalGraphBuffer(window_size=3, max_nodes=10)
        for _ in range(5):
            buf.push(_make_graph(4))
        assert len(buf.buffer) == 3


# ===========================================================
# RealtimeGraphBuilder Tests (import from pipeline)
# ===========================================================

# We import the builder from the pipeline module
sys.path.insert(0, str(_src_dir))
from run_pipeline_realtime import RealtimeGraphBuilder, GRAPH_K


class _FakePerson:
    """Minimal stand-in for TrackedPerson."""
    def __init__(self, track_id, cx, cy, x1=0, y1=0, x2=10, y2=10, conf=0.9):
        self.track_id = track_id
        self.cx = cx
        self.cy = cy
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf


class TestGraphBuilder:
    """Test kNN graph construction, features, and masking."""

    def _persons(self, n: int, spread: float = 100.0):
        """Create n fake persons in a 640×480 frame."""
        return [
            _FakePerson(
                track_id=i,
                cx=50 + i * spread,
                cy=50 + i * spread * 0.5,
                x1=40 + i * spread,
                y1=40 + i * spread * 0.5,
                x2=60 + i * spread,
                y2=60 + i * spread * 0.5,
            )
            for i in range(n)
        ]

    def test_knn_edges_created(self):
        """Graph should have kNN edges, not radius edges."""
        builder = RealtimeGraphBuilder(k=3, max_nodes=20)
        persons = self._persons(6, spread=50)
        graph = builder.build_graph(persons, (480, 640), {})

        assert graph is not None
        edge_index = graph["edge_index"]
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0, "kNN graph should produce edges"

    def test_k_effective_small_crowd(self):
        """With only 3 people and k=5, k_effective should be 2."""
        builder = RealtimeGraphBuilder(k=5, max_nodes=20)
        persons = self._persons(3, spread=50)
        graph = builder.build_graph(persons, (480, 640), {})

        assert graph is not None
        edge_index = graph["edge_index"]
        # Each node should have at most 2 neighbors
        for i in range(3):
            outgoing = np.sum(edge_index[0] == i)
            assert outgoing <= 2, f"Node {i} has {outgoing} neighbors but max is 2"

    def test_no_edges_to_padded_nodes(self):
        """Edges must only involve real nodes (indices < n_actual)."""
        builder = RealtimeGraphBuilder(k=3, max_nodes=20)
        persons = self._persons(5, spread=30)
        graph = builder.build_graph(persons, (480, 640), {})

        edge_index = graph["edge_index"]
        assert np.all(edge_index < 5), "No edge should reference a padded node index"

    def test_eight_features(self):
        """Node features should be 8-dimensional."""
        builder = RealtimeGraphBuilder(k=3, max_nodes=20)
        persons = self._persons(4, spread=40)
        graph = builder.build_graph(persons, (480, 640), {})

        assert graph["x"].shape == (20, 8)  # padded to max_nodes=20

    def test_mask_correctness(self):
        """Mask should mark exactly the real nodes."""
        builder = RealtimeGraphBuilder(k=3, max_nodes=20)
        persons = self._persons(7, spread=30)
        graph = builder.build_graph(persons, (480, 640), {})

        mask = graph["mask"]
        assert mask.shape == (20,)
        assert np.sum(mask) == 7
        np.testing.assert_array_equal(mask[:7], 1.0)
        np.testing.assert_array_equal(mask[7:], 0.0)

    def test_velocity_clipping(self):
        """Large position jumps should be clipped."""
        builder = RealtimeGraphBuilder(k=2, max_nodes=10)
        persons = self._persons(3, spread=20)

        # Simulate huge jump: prev position was very far
        prev = {0: (0.0, 0.0), 1: (1.0, 1.0), 2: (0.5, 0.5)}
        graph = builder.build_graph(persons, (480, 640), prev)

        # dx, dy are at columns 2, 3
        dx = graph["x"][:3, 2]
        dy = graph["x"][:3, 3]
        from run_pipeline_realtime import MAX_VELOCITY
        assert np.all(np.abs(dx) <= MAX_VELOCITY + 1e-6)
        assert np.all(np.abs(dy) <= MAX_VELOCITY + 1e-6)

    def test_returns_none_below_min_nodes(self):
        """Should return None with fewer than MIN_NODES persons."""
        builder = RealtimeGraphBuilder(k=3, max_nodes=10)
        graph = builder.build_graph(self._persons(1), (480, 640), {})
        assert graph is None

    def test_density_scale_invariant(self):
        """local_density should be degree / k (not degree / N-1)."""
        builder = RealtimeGraphBuilder(k=3, max_nodes=20)
        persons = self._persons(6, spread=20)
        graph = builder.build_graph(persons, (480, 640), {})
        density_col = graph["x"][:6, 6]  # density at feature index 6
        assert np.all(density_col >= 0)
        assert np.all(density_col <= 2.0)  # symmetric kNN can give up to ~2


# ===========================================================
# CrowdMetrics Tests
# ===========================================================

class TestCrowdMetrics:
    """Test mask-aware metrics computation."""

    def test_mask_filters_padded_nodes(self):
        """Metrics should only use unmasked nodes."""
        x = np.zeros((10, 8), dtype=np.float32)
        # Set real nodes (first 3) to have speed
        x[0, 2] = 0.1  # dx
        x[1, 2] = 0.2
        x[2, 2] = 0.3
        mask = np.zeros(10, dtype=np.float32)
        mask[:3] = 1.0

        metrics = CrowdMetrics.compute({"x": x, "mask": mask})
        assert metrics["mean_speed"] > 0, "Should compute speed from real nodes"

    def test_auto_detect_legacy_features(self):
        """5-feature graph should auto-use legacy FeatureIndex."""
        x = np.random.randn(5, 5).astype(np.float32)
        x[:, 4] = 0.5  # density at column 4
        metrics = CrowdMetrics.compute({"x": x})
        assert metrics["mean_density"] == pytest.approx(0.5)

    def test_auto_detect_expanded_features(self):
        """8-feature graph should use expanded FeatureIndex (density at col 6)."""
        x = np.zeros((5, 8), dtype=np.float32)
        x[:, 6] = 0.7  # density at column 6
        metrics = CrowdMetrics.compute({"x": x})
        assert metrics["mean_density"] == pytest.approx(0.7)

    def test_none_graph(self):
        """None graph should return empty metrics."""
        metrics = CrowdMetrics.compute(None)
        assert metrics["mean_speed"] == 0.0


# ===========================================================
# Integration Smoke Test
# ===========================================================

class TestIntegration:
    """Smoke test: temporal buffer + graph builder produce valid output."""

    def test_full_pipeline_flow(self):
        """3 frames through builder + buffer should produce valid x_seq."""
        builder = RealtimeGraphBuilder(k=3, max_nodes=15)
        buf = TemporalGraphBuffer(window_size=3, max_nodes=15)

        prev_pos = {}
        for frame_i in range(3):
            persons = [
                _FakePerson(
                    track_id=j,
                    cx=100 + j * 30 + frame_i * 2,
                    cy=100 + j * 20 + frame_i * 1,
                )
                for j in range(5 + frame_i)  # varying person count!
            ]
            graph = builder.build_graph(persons, (480, 640), prev_pos)
            assert graph is not None

            prev_pos = {
                p.track_id: (p.cx / 640, p.cy / 480)
                for p in persons
            }

            x_seq, mask_seq = buf.push(graph)

        # After 3 pushes with varying node counts, buffer should be ready
        assert x_seq is not None, "Buffer must fill even with varying node counts"
        assert x_seq.shape == (1, 3, 15, 8)
        assert mask_seq.shape == (1, 3, 15)

        # Verify anomaly score can be computed (just check shape/type)
        assert isinstance(x_seq, np.ndarray)
        assert x_seq.dtype == np.float32

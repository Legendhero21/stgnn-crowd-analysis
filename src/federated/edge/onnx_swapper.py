"""
ONNX Model Hot-Swap
-------------------
Thread-safe ONNX model replacement for continuous inference.

Key design:
- Inference continues uninterrupted during model swap
- New session loaded in background before swap
- Atomic reference swap under lock
- Old session cleaned up by GC

This allows federated learning updates without stopping inference.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import onnxruntime as ort


logger = logging.getLogger(__name__)


@dataclass
class ONNXModelInfo:
    """Metadata about an ONNX model."""
    path: str
    version: int
    input_names: List[str]
    output_names: List[str]
    loaded_at: float  # Unix timestamp
    
    def __str__(self) -> str:
        return f"ONNX(v{self.version}, {Path(self.path).name})"


class ONNXHotSwapper:
    """
    Thread-safe ONNX model hot-swapping for continuous inference.
    
    Guarantees:
    - Inference is never blocked during model load
    - Atomic model reference swap
    - Version tracking for federated sync
    - Graceful fallback on load failure
    
    Usage:
        swapper = ONNXHotSwapper("initial_model.onnx")
        
        # Inference (can be called from multiple threads)
        outputs = swapper.run(x_seq, edge_index)
        
        # Hot-swap (non-blocking for inference)
        swapper.hot_swap("new_model.onnx", version=2)
    """
    
    def __init__(
        self,
        initial_model_path: str,
        initial_version: int = 0,
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,
    ):
        """
        Initialize with an initial ONNX model.
        
        Args:
            initial_model_path: Path to initial ONNX model.
            initial_version: Initial model version number.
            providers: ONNX Runtime execution providers.
            session_options: ONNX Runtime session options.
        
        Raises:
            FileNotFoundError: If initial model doesn't exist.
            RuntimeError: If initial model fails to load.
        """
        if not os.path.isfile(initial_model_path):
            raise FileNotFoundError(f"Initial ONNX model not found: {initial_model_path}")
        
        self._providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session_options = session_options
        
        # Lock for atomic session swap (not for inference)
        self._swap_lock = threading.RLock()
        
        # Current session - accessed atomically
        self._session: Optional[ort.InferenceSession] = None
        self._model_info: Optional[ONNXModelInfo] = None
        
        # Load initial model
        self._load_session(initial_model_path, initial_version)
        
        logger.info("ONNXHotSwapper initialized with %s", self._model_info)
    
    def _load_session(self, model_path: str, version: int) -> ort.InferenceSession:
        """
        Load an ONNX session (internal, not thread-safe).
        
        Args:
            model_path: Path to ONNX model.
            version: Model version number.
        
        Returns:
            Loaded InferenceSession.
        
        Raises:
            RuntimeError: If loading fails.
        """
        try:
            session = ort.InferenceSession(
                model_path,
                providers=self._providers,
                sess_options=self._session_options,
            )
        except Exception as exc:
            logger.error("Failed to load ONNX model %s: %s", model_path, exc)
            raise RuntimeError(f"ONNX load failed: {exc}") from exc
        
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        
        model_info = ONNXModelInfo(
            path=model_path,
            version=version,
            input_names=input_names,
            output_names=output_names,
            loaded_at=time.time(),
        )
        
        # Atomic swap
        with self._swap_lock:
            self._session = session
            self._model_info = model_info
        
        return session
    
    def hot_swap(
        self,
        new_model_path: str,
        new_version: int,
        validate: bool = True,
        rollback_on_failure: bool = True,
    ) -> bool:
        """
        Hot-swap the ONNX model atomically.
        
        The new model is loaded in the background. If loading succeeds,
        the session reference is swapped atomically. Ongoing inference
        calls complete with the old model; new calls use the new model.
        
        Args:
            new_model_path: Path to new ONNX model.
            new_version: New model version number.
            validate: If True, run validation inference before swap.
            rollback_on_failure: If True, keep old model on any failure.
        
        Returns:
            True if swap succeeded, False otherwise.
        """
        if not os.path.isfile(new_model_path):
            logger.error("New model file not found: %s", new_model_path)
            return False
        
        logger.info("Starting hot-swap to version %d: %s", new_version, new_model_path)
        
        # Save old session for rollback
        old_session = self._session
        old_info = self._model_info
        
        try:
            # Step 1: Load new session (outside lock - may take time)
            new_session = ort.InferenceSession(
                new_model_path,
                providers=self._providers,
                sess_options=self._session_options,
            )
            
            input_names = [inp.name for inp in new_session.get_inputs()]
            output_names = [out.name for out in new_session.get_outputs()]
            
            # Step 2: Validate new session (optional)
            if validate:
                if not self._validate_session(new_session, input_names):
                    raise RuntimeError("New model failed validation")
            
            # Step 3: Create new info
            new_info = ONNXModelInfo(
                path=new_model_path,
                version=new_version,
                input_names=input_names,
                output_names=output_names,
                loaded_at=time.time(),
            )
            
            # Step 4: Atomic swap under lock
            with self._swap_lock:
                self._session = new_session
                self._model_info = new_info
            
            logger.info(
                "Hot-swap complete: v%d -> v%d (took %.1fms)",
                old_info.version if old_info else 0,
                new_version,
                (new_info.loaded_at - (old_info.loaded_at if old_info else new_info.loaded_at)) * 1000,
            )
            
            # Old session will be garbage collected
            del old_session
            
            return True
            
        except Exception as exc:
            logger.error("Hot-swap failed: %s", exc)
            
            if rollback_on_failure and old_session is not None:
                logger.info("Rolling back to version %d", old_info.version if old_info else 0)
                with self._swap_lock:
                    self._session = old_session
                    self._model_info = old_info
            
            return False
    
    def _validate_session(
        self,
        session: ort.InferenceSession,
        input_names: List[str],
    ) -> bool:
        """
        Validate a session by running dummy inference.
        
        Args:
            session: ONNX session to validate.
            input_names: Expected input names.
        
        Returns:
            True if validation passed.
        """
        try:
            # Create dummy inputs matching STGNN expected shapes
            # x: [1, T, N, F] where T=5, N=10, F=5
            # edge_index: [2, E] where E=20
            dummy_x = np.random.randn(1, 5, 10, 5).astype(np.float32)
            dummy_edges = np.random.randint(0, 10, (2, 20)).astype(np.int64)
            
            inputs = {}
            for name in input_names:
                if "edge" in name.lower() or "index" in name.lower():
                    inputs[name] = dummy_edges
                else:
                    inputs[name] = dummy_x
            
            # Run inference
            outputs = session.run(None, inputs)
            
            if outputs is None or len(outputs) == 0:
                logger.warning("Validation produced no outputs")
                return False
            
            # Check output has valid shape
            out = outputs[0]
            if out is None or out.size == 0:
                logger.warning("Validation output is empty")
                return False
            
            logger.debug("Validation passed: output shape %s", out.shape)
            return True
            
        except Exception as exc:
            logger.warning("Validation inference failed: %s", exc)
            return False
    
    def run(
        self,
        x_seq: np.ndarray,
        edge_index: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Run inference on the current model.
        
        Thread-safe. Multiple calls can run concurrently.
        
        Args:
            x_seq: Input features [1, T, N, F].
            edge_index: Graph edges [2, E].
        
        Returns:
            Tuple of (predictions, anomaly_score).
            predictions: [N, 2] or [1, N, 2] predicted positions.
            anomaly_score: MSE between prediction and last frame positions.
        
        Raises:
            RuntimeError: If no model is loaded.
        """
        # Get session reference (atomic read, no lock needed)
        session = self._session
        info = self._model_info
        
        if session is None or info is None:
            raise RuntimeError("No ONNX model loaded")
        
        # Validate inputs
        if x_seq is None or x_seq.size == 0:
            return np.empty((0, 2), dtype=np.float32), 0.0
        
        if edge_index is None or edge_index.size == 0:
            return np.empty((0, 2), dtype=np.float32), 0.0
        
        # Prepare inputs
        x_seq = x_seq.astype(np.float32, copy=False)
        edge_index = edge_index.astype(np.int64, copy=False)
        
        inputs = {}
        for name in info.input_names:
            if "edge" in name.lower() or "index" in name.lower():
                inputs[name] = edge_index
            else:
                inputs[name] = x_seq
        
        # Run inference
        try:
            outputs = session.run(info.output_names, inputs)
        except Exception as exc:
            logger.error("ONNX inference failed: %s", exc)
            return np.empty((0, 2), dtype=np.float32), 0.0
        
        # Process outputs
        preds = np.asarray(outputs[0], dtype=np.float32)
        
        # Handle shape: expected [1, N, 2] or [N, 2]
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]  # [N, 2]
        elif preds.ndim != 2:
            try:
                preds = preds.reshape(-1, 2)
            except Exception:
                return preds, 0.0
        
        # Compute anomaly score (MSE vs last frame positions)
        if x_seq.ndim == 4 and x_seq.shape[0] == 1:
            last_positions = x_seq[0, -1, :, :2]  # [N, 2]
            
            if preds.shape[0] == last_positions.shape[0]:
                mse = float(np.mean((preds[:, :2] - last_positions) ** 2))
            else:
                mse = 0.0
        else:
            mse = 0.0
        
        return preds, mse
    
    def predict_from_sequence(
        self,
        x_seq: np.ndarray,
        edge_index: np.ndarray,
    ) -> float:
        """
        Compatibility method matching original STGNNInference interface.
        
        Args:
            x_seq: Input features [1, T, N, F].
            edge_index: Graph edges [2, E].
        
        Returns:
            Anomaly score (float).
        """
        _, anomaly_score = self.run(x_seq, edge_index)
        return anomaly_score
    
    @property
    def version(self) -> int:
        """Current model version."""
        return self._model_info.version if self._model_info else 0
    
    @property
    def model_path(self) -> str:
        """Current model path."""
        return self._model_info.path if self._model_info else ""
    
    @property
    def info(self) -> Optional[ONNXModelInfo]:
        """Current model info."""
        return self._model_info
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get model statistics.
        
        Returns:
            Dict with version, path, uptime, etc.
        """
        info = self._model_info
        if info is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "version": info.version,
            "path": info.path,
            "input_names": info.input_names,
            "output_names": info.output_names,
            "uptime_sec": time.time() - info.loaded_at,
        }

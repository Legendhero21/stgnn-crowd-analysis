# Federated Client Package
"""
Client-side components for federated learning.

This package provides:
- LocalTrainer: PyTorch training loop for edge devices
- FederatedClient: Orchestrates edge + training + server communication
"""

from .local_trainer import LocalTrainer, TrainingResult
from .federated_client import FederatedClient, FederatedClientConfig

__all__ = [
    "LocalTrainer",
    "TrainingResult",
    "FederatedClient",
    "FederatedClientConfig",
]

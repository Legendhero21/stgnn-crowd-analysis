# Federated Transport Layer
"""
Transport adapters for federated learning communication.

This package provides transport abstractions that allow swapping
between local (in-process) and network-based communication.

Available:
- LocalTransport: In-process calls for testing and single-machine setups
- (Future) SocketTransport: TCP-based for multi-machine deployments
"""

from .local_transport import LocalTransport, TransportProtocol

__all__ = [
    "LocalTransport",
    "TransportProtocol",
]

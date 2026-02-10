# Federated STGNN
"""
Federated learning system for distributed STGNN crowd analysis.

Subpackages:
- edge: Edge device components (EdgeClient, VideoSource, etc.)
- server: Central server components (FederatedServer, Aggregator, etc.)
- protocol: Communication protocol and message types
- transport: Transport adapters (LocalTransport, etc.)
- client: Client-side federated components (FederatedClient, LocalTrainer)
"""

# Version
__version__ = "0.1.0"

# Subpackage imports for convenience
from . import edge
from . import server
from . import protocol
from . import transport
from . import client

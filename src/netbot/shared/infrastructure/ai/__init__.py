"""
AI infrastructure for NetBot V2.
"""

from .model_client import ModelClient, get_model_client
from .embedding_client import EmbeddingClient, get_embedding_client

__all__ = [
    "ModelClient",
    "get_model_client",
    "EmbeddingClient", 
    "get_embedding_client",
]
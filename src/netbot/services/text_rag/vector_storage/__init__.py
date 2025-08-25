"""
Vector storage backends for TextRAG.

Provides pluggable vector database backends for embedding storage
and similarity search operations.
"""

from .base import VectorStore
from .chroma_store import ChromaVectorStore
from .memory_store import InMemoryVectorStore

__all__ = [
    'VectorStore',
    'ChromaVectorStore', 
    'InMemoryVectorStore'
]
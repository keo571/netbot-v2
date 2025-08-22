"""
Embeddings module for NetBot-v2.

Core functionality (always available):
- EmbeddingEncoder: Compute embeddings for graph nodes  
- EmbeddingManager: Add embeddings to Neo4j graphs

Advanced functionality (requires optional dependencies):
- Located in embeddings.advanced submodule
- Includes hybrid RAG features, document chunking, vector stores
"""

# Core embeddings (always available)
from .embedding_encoder import EmbeddingEncoder
from .client import EmbeddingManager

# Core functionality is always available
__all__ = [
    'EmbeddingEncoder',         # Core: compute embeddings
    'EmbeddingManager'          # Core: manage Neo4j embeddings  
]

# Advanced features are available via embeddings.advanced submodule
# Example: from embeddings.advanced import HybridEmbeddingManager
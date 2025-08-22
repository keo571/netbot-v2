"""
GraphRAG: Retrieval-Augmented Generation over Knowledge Graphs

A modular system for semantic search and visualization over graph data.
"""

from .client import GraphRAG
from models.graph_models import GraphNode, GraphRelationship

__version__ = "1.0.0"
__all__ = ["GraphRAG", "GraphNode", "GraphRelationship"]
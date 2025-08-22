"""
Shared data models for the entire netbot-v2 system.

Contains canonical definitions for graph entities used across:
- diagram_processing (data creation)
- graph_rag (data consumption)
- Any other components that work with graph data
"""

from .graph_models import GraphNode, GraphRelationship, Shape

__all__ = ['GraphNode', 'GraphRelationship', 'Shape']
"""
NetBot V2 - Advanced AI-powered diagram analysis and hybrid RAG system.
"""

__version__ = "2.0.0"
__author__ = "NetBot Team"

# Re-export main components for easy access
from .shared.config.settings import get_settings
from .shared.models.graph import GraphNode, GraphRelationship
from .shared.exceptions import NetBotError, ConfigurationError

__all__ = [
    "get_settings",
    "GraphNode", 
    "GraphRelationship",
    "NetBotError",
    "ConfigurationError",
]
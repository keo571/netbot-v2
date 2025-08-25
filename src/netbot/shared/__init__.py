"""
Shared components for NetBot V2.

Contains common models, utilities, and infrastructure used across all services.
This package provides the foundation for all NetBot services with:

- Common data models and validation
- Centralized configuration management  
- Shared exception hierarchy
- Infrastructure services (database, cache, AI, monitoring)
"""

from .models import *
from .config import *
from .exceptions import *
from .infrastructure import *

__all__ = [
    # From models
    "BaseModel", "TimestampMixin", "MetadataMixin",
    "GraphNode", "GraphRelationship", "GraphResult", "Shape",
    "Query", "SearchResult", "Context", "RetrievalMetadata",
    
    # From config
    "Settings", "get_settings",
    
    # From exceptions
    "NetBotError", "ConfigurationError", "DatabaseError", 
    "ProcessingError", "ValidationError", "AIError", "CacheError",
    
    # From infrastructure
    "DatabaseManager", "get_database", "BaseRepository",
    "CacheManager", "get_cache_manager",
    "ModelClient", "get_model_client",
    "EmbeddingClient", "get_embedding_client", 
    "get_logger", "setup_logging", "MetricsCollector", "get_metrics",
]
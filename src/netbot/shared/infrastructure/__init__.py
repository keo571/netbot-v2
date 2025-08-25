"""
Shared infrastructure components for NetBot V2.

Provides centralized infrastructure services including:
- Database connection management and pooling
- Advanced caching with multiple backends
- AI model clients with caching and rate limiting
- Embedding services with batch processing
- Monitoring and metrics collection
"""

from .database.connection_manager import DatabaseManager, get_database
from .database.base_repository import BaseRepository
from .cache.cache_manager import CacheManager, get_cache_manager
from .ai.model_client import ModelClient, get_model_client
from .ai.embedding_client import EmbeddingClient, get_embedding_client
from .monitoring.logger import get_logger, setup_logging
from .monitoring.metrics import MetricsCollector, get_metrics

__all__ = [
    # Database
    "DatabaseManager",
    "get_database",
    "BaseRepository",
    
    # Cache
    "CacheManager", 
    "get_cache_manager",
    
    # AI Services
    "ModelClient",
    "get_model_client",
    "EmbeddingClient",
    "get_embedding_client",
    
    # Monitoring
    "get_logger",
    "setup_logging",
    "MetricsCollector",
    "get_metrics",
]
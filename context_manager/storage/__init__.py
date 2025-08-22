"""
Storage backends for context management.

Provides multiple storage implementations for different deployment scenarios:
- In-memory stores: No dependencies (development/testing)
- Redis: High-performance session storage (optional: redis)
- MongoDB: Document-based history storage (optional: pymongo) 
- PostgreSQL: Relational user storage (optional: psycopg2)

Usage:
    # Core (no dependencies)
    from context_manager.storage import InMemorySessionStore
    
    # With optional dependencies
    from context_manager.storage import RedisSessionStore, MongoHistoryStore
"""

# Try to import all storage backends
try:
    from .backends import (
        # Abstract base classes
        SessionStore, HistoryStore, UserStore,
        
        # In-memory implementations (no dependencies)
        InMemorySessionStore, InMemoryHistoryStore, InMemoryUserStore,
        
        # External storage backends (optional dependencies)
        RedisSessionStore, MongoHistoryStore, PostgreSQLUserStore
    )
    
    __all__ = [
        # Abstract interfaces
        'SessionStore', 'HistoryStore', 'UserStore',
        
        # In-memory (always available)
        'InMemorySessionStore', 'InMemoryHistoryStore', 'InMemoryUserStore',
        
        # External backends (optional)
        'RedisSessionStore', 'MongoHistoryStore', 'PostgreSQLUserStore'
    ]
    
    EXTERNAL_STORAGE_AVAILABLE = True
    
except ImportError as e:
    # External storage dependencies not available
    from .backends import (
        SessionStore, HistoryStore, UserStore,
        InMemorySessionStore, InMemoryHistoryStore, InMemoryUserStore
    )
    
    __all__ = [
        'SessionStore', 'HistoryStore', 'UserStore',
        'InMemorySessionStore', 'InMemoryHistoryStore', 'InMemoryUserStore'
    ]
    
    EXTERNAL_STORAGE_AVAILABLE = False
    _IMPORT_ERROR = str(e)

def check_external_storage():
    """Check if external storage backends are available."""
    if EXTERNAL_STORAGE_AVAILABLE:
        return True, "External storage backends available"
    else:
        return False, f"External storage not available: {_IMPORT_ERROR}"
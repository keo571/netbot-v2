"""
Context Manager for stateful RAG chatbot systems.

This package provides components for maintaining conversational state,
user preferences, and conversation history in AI-powered applications.

Core components (always available):
- ContextManager: Main client interface
- Models: Data structures for sessions, conversations, preferences
- Core processing: PromptBuilder, QueryRewriter, RetrievalFilter
- In-memory storage: For development and testing

Optional components (require external dependencies):
- External storage: Redis, MongoDB, PostgreSQL backends
- Advanced analytics and migration tools

Usage:
    # Basic usage (no external dependencies)
    from context_manager import ContextManager, InMemorySessionStore
    
    # With external storage
    from context_manager.storage import RedisSessionStore, MongoHistoryStore
"""

# Core components (always available)
from .models import (
    SessionState, ConversationHistory, UserPreferences, ConversationExchange,
    QueryIntent, ResponseStyle, ExpertiseLevel, UserFeedback
)
from .client import ContextManager
from .config import Config, load_config, create_example_env_file

# Core functionality
from .core import PromptBuilder, QueryRewriter, RetrievalFilter

# Basic storage (no external dependencies)
from .storage import (
    SessionStore, HistoryStore, UserStore,
    InMemorySessionStore, InMemoryHistoryStore, InMemoryUserStore
)

# Utilities
from .utils import (
    generate_session_id, calculate_similarity, truncate_text, sanitize_user_input
)

# Core exports (always available)
__all__ = [
    # Models
    "SessionState", "ConversationHistory", "UserPreferences", "ConversationExchange",
    "QueryIntent", "ResponseStyle", "ExpertiseLevel", "UserFeedback",
    
    # Main client
    "ContextManager",
    
    # Configuration
    "Config", "load_config", "create_example_env_file",
    
    # Core processing
    "PromptBuilder", "QueryRewriter", "RetrievalFilter",
    
    # Basic storage interfaces
    "SessionStore", "HistoryStore", "UserStore",
    "InMemorySessionStore", "InMemoryHistoryStore", "InMemoryUserStore",
    
    # Utilities
    "generate_session_id", "calculate_similarity", "truncate_text", "sanitize_user_input"
]

# External storage and advanced features available via submodules:
# - context_manager.storage: All storage backends (including external)
# - context_manager.utils: Analytics, maintenance, migration tools
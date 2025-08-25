"""
Context Manager Service for NetBot V2.

Provides stateful conversation capabilities for RAG systems including:
- Session management and conversation history
- User preference tracking and learning
- Context-aware query enhancement
- Contextual response filtering and ranking
"""

from .service import ContextManagerService
from .models import Session, Message, User, ContextConfig
from .client import ContextManager

__all__ = [
    "ContextManagerService",
    "ContextManager", 
    "Session",
    "Message",
    "User",
    "ContextConfig",
]
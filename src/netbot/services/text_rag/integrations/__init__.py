"""
Integration modules for TextRAG.

Provides integration with other NetBot V2 services including
Context Manager for conversational capabilities.
"""

from .context_integration import ContextAwareTextRAG

__all__ = [
    'ContextAwareTextRAG'
]
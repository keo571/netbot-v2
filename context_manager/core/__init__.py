"""
Core context management functionality.

These components provide the essential features for conversational AI context:
- PromptBuilder: Dynamic prompt construction
- QueryRewriter: Query enhancement and rewriting  
- RetrievalFilter: Context-aware result filtering

Dependencies: Only core Python libraries (no external databases required)
"""

from .prompt_builder import PromptBuilder
from .query_rewriter import QueryRewriter
from .retrieval_filter import RetrievalFilter

__all__ = [
    'PromptBuilder',      # Dynamic prompt construction
    'QueryRewriter',      # Query enhancement 
    'RetrievalFilter'     # Context-aware filtering
]
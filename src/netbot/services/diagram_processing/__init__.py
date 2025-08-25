"""
Diagram Processing Service for NetBot V2.

Provides a clean service layer for converting diagrams into knowledge graphs.
Uses the shared infrastructure for database, caching, and AI operations.
"""

from .service import DiagramProcessingService
from .models import ProcessingRequest, ProcessingResult

__all__ = [
    "DiagramProcessingService",
    "ProcessingRequest",
    "ProcessingResult",
]
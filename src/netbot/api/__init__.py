"""
NetBot V2 API Gateway.

Provides unified REST API access to all NetBot services with proper
error handling, monitoring, and documentation.
"""

from .app import create_app
from .models import APIResponse, ErrorResponse

__all__ = [
    "create_app",
    "APIResponse",
    "ErrorResponse",
]
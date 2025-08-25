"""
API models for request/response handling.
"""

from typing import Any, Optional, Dict, List
from pydantic import Field

from ..shared.models.base import BaseModel


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    
    success: bool = Field(..., description="Whether the request succeeded")
    data: Any = Field(default=None, description="Response data")
    message: Optional[str] = Field(default=None, description="Response message")
    timestamp: Optional[str] = Field(default=None, description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": {"result": "example data"},
                "message": "Operation completed successfully",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: Optional[str] = Field(default=None, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "ValidationError",
                "message": "Invalid input parameters",
                "details": {"field": "diagram_id", "issue": "required field missing"},
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    services: Dict[str, str] = Field(default_factory=dict, description="Individual service status")
    timestamp: str = Field(..., description="Check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "services": {
                    "database": "connected",
                    "embedding_client": "loaded",
                    "model_client": "ready"
                },
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
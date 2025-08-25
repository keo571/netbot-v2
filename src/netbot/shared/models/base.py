"""
Base models and mixins for NetBot V2.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel as PydanticBaseModel, Field


class BaseModel(PydanticBaseModel):
    """
    Base model for all NetBot data structures.
    
    Provides common configuration and utilities.
    """
    
    class Config:
        # Allow field population by name or alias
        validate_by_name = True
        # Validate assignments after object creation
        validate_assignment = True
        # Use enum values instead of enum names
        use_enum_values = True
        # Allow extra fields (for extensibility)
        extra = "forbid"


class TimestampMixin:
    """
    Mixin to add timestamp fields to models.
    """
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    
    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


class MetadataMixin:
    """
    Mixin to add metadata field to models.
    """
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add a metadata entry."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value."""
        return self.metadata.get(key, default)
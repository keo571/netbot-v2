"""
Service models for diagram processing operations.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import Field, validator

from ...shared.models.base import BaseModel, TimestampMixin
from ...shared.models.graph import GraphNode, GraphRelationship


class ProcessingRequest(BaseModel):
    """Request for diagram processing."""
    
    image_path: str = Field(..., description="Path to the image file")
    diagram_id: str = Field(..., description="Unique diagram identifier")
    output_dir: Optional[str] = Field(default=None, description="Output directory for results")
    
    # Processing options
    ocr_enabled: bool = Field(default=True, description="Enable OCR processing")
    shape_detection: bool = Field(default=True, description="Enable shape detection")
    store_in_database: bool = Field(default=True, description="Store results in Neo4j")
    
    # AI model settings
    model_name: str = Field(default="gemini-2.0-flash-exp", description="AI model to use")
    max_tokens: int = Field(default=8192, description="Maximum response tokens")
    temperature: float = Field(default=0.1, description="AI generation temperature")
    
    @validator('image_path')
    def validate_image_path(cls, v):
        """Validate image path exists."""
        if not Path(v).exists():
            raise ValueError(f"Image file does not exist: {v}")
        return v
    
    @validator('diagram_id')
    def validate_diagram_id(cls, v):
        """Validate diagram ID format."""
        if not v or not v.strip():
            raise ValueError("Diagram ID cannot be empty")
        return v.strip()


class ProcessingResult(BaseModel, TimestampMixin):
    """Result of diagram processing operation."""
    
    # Request reference
    request: ProcessingRequest = Field(..., description="Original processing request")
    
    # Processing results
    nodes: List[GraphNode] = Field(default_factory=list, description="Extracted nodes")
    relationships: List[GraphRelationship] = Field(default_factory=list, description="Extracted relationships")
    
    # Processing metadata
    success: bool = Field(default=False, description="Whether processing succeeded")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Performance metrics
    total_duration_seconds: float = Field(default=0.0, description="Total processing time")
    ocr_duration_seconds: float = Field(default=0.0, description="OCR processing time")
    ai_duration_seconds: float = Field(default=0.0, description="AI processing time")
    storage_duration_seconds: float = Field(default=0.0, description="Storage time")
    
    # Output files (if saved)
    output_files: Dict[str, str] = Field(default_factory=dict, description="Generated output files")
    
    # Quality metrics
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall processing confidence")
    nodes_detected: int = Field(default=0, description="Number of nodes detected")
    relationships_detected: int = Field(default=0, description="Number of relationships detected")
    
    @property
    def node_count(self) -> int:
        """Number of successfully extracted nodes."""
        return len(self.nodes)
    
    @property
    def relationship_count(self) -> int:
        """Number of successfully extracted relationships."""
        return len(self.relationships)
    
    @property
    def is_empty(self) -> bool:
        """Check if no data was extracted."""
        return self.node_count == 0 and self.relationship_count == 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the processing results."""
        return {
            'diagram_id': self.request.diagram_id,
            'success': self.success,
            'node_count': self.node_count,
            'relationship_count': self.relationship_count,
            'confidence_score': self.confidence_score,
            'total_duration_seconds': self.total_duration_seconds,
            'created_at': self.created_at.isoformat(),
            'error_message': self.error_message
        }
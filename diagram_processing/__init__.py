"""
Diagram Processing: Convert network diagrams and flowcharts into knowledge graphs.

This package provides tools for:
- Processing diagram images with OCR and AI
- Extracting nodes and relationships  
- Storing results in CSV and Neo4j formats
"""

from .client import DiagramProcessor, process_diagram
from .core.pipeline import KnowledgeGraphPipeline
from models.graph_models import GraphNode, GraphRelationship, Shape

__version__ = "1.0.0"
__all__ = ["DiagramProcessor", "process_diagram", "KnowledgeGraphPipeline", "GraphNode", "GraphRelationship", "Shape"]
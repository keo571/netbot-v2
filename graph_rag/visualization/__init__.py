"""
Graph visualization package for netbot-v2.

Simple GraphViz-based visualization.

Usage:
    from graph_rag.visualization import GraphVisualizer
    
    # Create visualizer
    viz = GraphVisualizer()
    
    # Generate visualization
    viz.generate_image(nodes, relationships, 'output.png')
    
    # Generate base64 for web
    base64_img = viz.generate_image_base64(nodes, relationships)
"""

# Direct GraphViz visualizer
from .graphviz import GraphvizVisualizer as GraphVisualizer

# Base classes (for extension)
from .base import BaseVisualizer, VisualizationConfig, PropertySummaryMixin

__all__ = [
    'GraphVisualizer',
    'BaseVisualizer',
    'VisualizationConfig', 
    'PropertySummaryMixin',
]

__version__ = '2.0.0'
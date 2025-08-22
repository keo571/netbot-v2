"""
Graph visualization package for netbot-v2.

This package provides flexible graph visualization capabilities with multiple backends:
- NetworkX + Matplotlib: Interactive visualizations, good for exploration
- Graphviz: High-quality publication-ready graphics

Usage:
    from graph_rag.visualization import GraphVisualizer
    
    # Auto-select best available backend
    viz = GraphVisualizer()
    
    # Or specify a backend
    viz = GraphVisualizer(backend='graphviz')
    
    # Generate visualization
    viz.generate_image(nodes, relationships, 'output.png')

Advanced Usage:
    from graph_rag.visualization import VisualizationFactory, NetworkXVisualizer, GraphvizVisualizer
    
    # Check what's available
    print(VisualizationFactory.get_available_backends())
    
    # Use specific backends directly
    networkx_viz = NetworkXVisualizer()
    graphviz_viz = GraphvizVisualizer()
"""

# Main public interface
from .factory import GraphVisualizer, VisualizationFactory

# Backend-specific visualizers (for advanced usage)
from .networkx_viz import NetworkXVisualizer
from .graphviz_viz import GraphvizVisualizer

# Base classes (for extension)
from .base import BaseVisualizer, VisualizationConfig, PropertySummaryMixin

__all__ = [
    # Main interface
    'GraphVisualizer',
    'VisualizationFactory',
    
    # Specific backends
    'NetworkXVisualizer', 
    'GraphvizVisualizer',
    
    # Base classes
    'BaseVisualizer',
    'VisualizationConfig',
    'PropertySummaryMixin',
    
]

# Version information
__version__ = '2.0.0'
__author__ = 'netbot-v2'

def list_backends():
    """List available visualization backends."""
    return VisualizationFactory.get_available_backends()
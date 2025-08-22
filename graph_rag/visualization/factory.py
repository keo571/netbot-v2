"""
Visualization factory for automatic backend selection and unified interface.

This module provides a unified interface to different visualization backends,
automatically selecting the best available backend or allowing manual selection.
"""

from typing import List, Optional
from models.graph_models import GraphNode, GraphRelationship
from .base import BaseVisualizer, PropertySummaryMixin
from .networkx_viz import NetworkXVisualizer
from .graphviz_viz import GraphvizVisualizer


class VisualizationFactory:
    """Factory for creating and managing visualization backends."""
    
    # Registry of all supported backends (ordered by preference)
    BACKEND_REGISTRY = {
        'graphviz': GraphvizVisualizer,
        'networkx': NetworkXVisualizer,
    }
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """Get list of available visualization backends.
        
        Returns:
            List of backend names that are available on this system
        """
        available = []
        
        for backend_name, backend_class in VisualizationFactory.BACKEND_REGISTRY.items():
            viz = backend_class()
            if viz.is_available():
                available.append(backend_name)
        
        return available
    
    @staticmethod
    def create_visualizer(backend: Optional[str] = None) -> BaseVisualizer:
        """Create a visualizer instance.
        
        Args:
            backend: Specific backend to use ('networkx', 'graphviz'), 
                    or None for automatic selection
                    
        Returns:
            Visualizer instance
            
        Raises:
            RuntimeError: If no backends are available or requested backend is unavailable
        """
        # Auto-select backend if none specified
        if backend is None:
            # Try backends in registry order
            for backend_name, backend_class in VisualizationFactory.BACKEND_REGISTRY.items():
                if backend_class().is_available():
                    backend = backend_name
                    print(f"Auto-selected visualization backend: {backend}")
                    return backend_class()
            
            # If no backend available
            raise RuntimeError("No visualization backends are available. "
                             "Please install matplotlib+networkx and/or graphviz.")
        
        # User specified a backend - create and validate availability
        try:
            backend_class = VisualizationFactory.BACKEND_REGISTRY[backend]
        except KeyError:
            raise RuntimeError(f"Unknown backend '{backend}'. "
                             f"Supported backends: {list(VisualizationFactory.BACKEND_REGISTRY.keys())}")
        
        visualizer = backend_class()
        if not visualizer.is_available():
            available = VisualizationFactory.get_available_backends()
            raise RuntimeError(f"Backend '{backend}' is not available (missing dependencies). "
                             f"Available backends: {available}")
        
        return visualizer
    


class GraphVisualizer(PropertySummaryMixin):
    """
    Unified graph visualizer with automatic backend selection.
    
    Automatically selects the best available visualization backend
    and provides a clean, unified interface for graph visualization.
    """
    
    def __init__(self, backend: Optional[str] = None):
        """Initialize the graph visualizer.
        
        Args:
            backend: Visualization backend to use ('networkx', 'graphviz'), 
                    or None for automatic selection
        """
        self._backend_impl = VisualizationFactory.create_visualizer(backend)
        self.backend = backend or self._detect_backend()
    
    def _detect_backend(self) -> str:
        """Detect which backend was auto-selected."""
        for backend_name, backend_class in VisualizationFactory.BACKEND_REGISTRY.items():
            if isinstance(self._backend_impl, backend_class):
                return backend_name
    
    # ========== Unified Interface ==========
    
    def generate_image(self, nodes: List[GraphNode], relationships: List[GraphRelationship],
                      output_path: str, **kwargs) -> str:
        """Generate visualization using the selected backend.
        
        Args:
            nodes: List of nodes to visualize
            relationships: List of relationships to visualize
            output_path: Output file path
            **kwargs: Backend-specific options
            
        Returns:
            Path to generated image file
        """
        return self._backend_impl.generate_image(nodes, relationships, output_path, **kwargs)
    
    def switch_backend(self, backend: str):
        """Switch to a different visualization backend.
        
        Args:
            backend: Backend name ('networkx', 'graphviz')
        """
        self._backend_impl = VisualizationFactory.create_visualizer(backend)
        self.backend = backend
        print(f"Switched to {backend} backend")
     
    
    # ========== Jupyter Integration ==========
    
    def show_interactive(self, nodes: List[GraphNode], relationships: List[GraphRelationship], **kwargs):
        """Display graph inline in Jupyter notebooks (NetworkX only).
        
        This method provides interactive visualization for Jupyter notebooks by displaying
        the graph directly without saving to file. Only works with NetworkX backend.
        
        Args:
            nodes: List of nodes to visualize
            relationships: List of relationships to visualize
            **kwargs: Additional arguments passed to NetworkX visualizer
        """
        if self.backend == 'networkx':
            self._backend_impl.show_interactive(nodes, relationships, **kwargs)
        else:
            # Switch to NetworkX temporarily for interactive display
            print("📊 Switching to NetworkX backend for interactive display")
            try:
                networkx_viz = VisualizationFactory.create_visualizer('networkx')
                networkx_viz.show_interactive(nodes, relationships, **kwargs)
            except Exception as e:
                print(f"❌ Interactive display not available: {e}")
    
    # ========== Extended Features ==========
    
    def generate_comparison(self, nodes: List[GraphNode], relationships: List[GraphRelationship],
                           output_base: str, backends: Optional[List[str]] = None) -> dict:
        """Generate visualizations using multiple backends for comparison.
        
        Args:
            nodes: List of nodes to visualize
            relationships: List of relationships to visualize
            output_base: Base name for output files
            backends: List of backends to use, or None for all available
            
        Returns:
            Dictionary mapping backend names to output file paths
        """
        if backends is None:
            backends = VisualizationFactory.get_available_backends()
        
        results = {}
        
        for backend in backends:
            try:
                viz = VisualizationFactory.create_visualizer(backend)
                output_path = f"{output_base}_{backend}"
                result_path = viz.generate_image(nodes, relationships, output_path)
                if result_path:
                    results[backend] = result_path
                    print(f"Generated {backend} visualization: {result_path}")
            except Exception as e:
                print(f"❌ Failed to generate {backend} visualization: {e}")
                results[backend] = None
        
        return results
    

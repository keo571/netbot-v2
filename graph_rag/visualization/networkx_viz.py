"""
NetworkX-based graph visualization using Matplotlib.

This module provides graph visualization using NetworkX for layout algorithms
and Matplotlib for rendering. It's suitable for interactive environments
and provides good control over styling and layouts.
"""

from typing import List, Tuple, Dict
from models.graph_models import GraphNode, GraphRelationship
from .base import BaseVisualizer, PropertySummaryMixin

# Lazy import of optional dependencies
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import traceback
    _DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    nx = None
    plt = None
    mpatches = None
    traceback = None
    _DEPENDENCIES_AVAILABLE = False
    _IMPORT_ERROR = str(e)


class NetworkXVisualizer(BaseVisualizer, PropertySummaryMixin):
    """Graph visualizer using NetworkX and Matplotlib backend.
    
    This visualizer provides interactive graph visualization using NetworkX for layout
    algorithms and Matplotlib for rendering. It supports multiple layout types and
    customizable styling options.
    """
    
    # Configuration constants
    DEFAULT_FIGURE_SIZE = (12, 8)
    LARGE_FIGURE_SIZE = (16, 12)  # For plots with properties
    DEFAULT_DPI = 100
    HIGH_DPI = 300  # For saved images
    DEFAULT_NODE_SIZE = 1000
    DEFAULT_EDGE_WIDTH = 2
    DEFAULT_FONT_SIZE = 8
    DEFAULT_LEGEND_FONT_SIZE = 10
    TITLE_FONT_SIZE = 16
    LAYOUT_SPRING_K = 2
    LAYOUT_ITERATIONS = 50
    
    def __init__(self):
        """Initialize NetworkX visualizer."""
        super().__init__()
        self._dependencies_available = _DEPENDENCIES_AVAILABLE
        if not _DEPENDENCIES_AVAILABLE:
            self._import_error = _IMPORT_ERROR
    
    def is_available(self) -> bool:
        """Check if NetworkX backend is available."""
        return self._dependencies_available
    
    # ========== Public Interface ==========
    
    def generate_image(self, nodes: List[GraphNode], relationships: List[GraphRelationship], 
                      output_path: str = "subgraph.png", layout: str = "spring",
                      show_node_properties: bool = True, show_edge_properties: bool = True) -> str:
        """Generate a graph image using NetworkX and Matplotlib.
        
        Args:
            nodes: List of nodes to visualize
            relationships: List of relationships to visualize
            output_path: Path for output image file
            layout: Layout algorithm ('spring', 'circular', 'shell', 'kamada_kawai', 'random')
            show_node_properties: Whether to show node properties in labels
            show_edge_properties: Whether to show edge properties in labels
            
        Returns:
            Path to generated image file, or empty string if failed
        """
        return self._create_visualization(nodes, relationships, layout, show_node_properties, 
                                        show_edge_properties, output_path=output_path)
    
    def show_interactive(self, nodes: List[GraphNode], relationships: List[GraphRelationship],
                        layout: str = "spring", show_node_properties: bool = True, 
                        show_edge_properties: bool = True):
        """Display an interactive graph visualization inline (for Jupyter notebooks).
        
        This method displays the graph directly in Jupyter notebooks without saving to file.
        Perfect for data exploration and iterative analysis.
        
        Args:
            nodes: List of nodes to visualize
            relationships: List of relationships to visualize
            layout: Layout algorithm ('spring', 'circular', 'shell', 'kamada_kawai', 'random')
            show_node_properties: Whether to show node properties in labels
            show_edge_properties: Whether to show edge properties in labels
        """
        self._create_visualization(nodes, relationships, layout, show_node_properties, 
                                 show_edge_properties, output_path=None)
    
    def generate_image_base64(self, nodes: List[GraphNode], relationships: List[GraphRelationship], 
                             layout: str = "spring", show_node_properties: bool = True, 
                             show_edge_properties: bool = True) -> str:
        """Generate a graph image as base64 string without saving file.
        
        Args:
            nodes: List of nodes to visualize
            relationships: List of relationships to visualize
            layout: Layout algorithm ('spring', 'circular', 'shell', 'kamada_kawai', 'random')
            show_node_properties: Whether to show node properties in labels
            show_edge_properties: Whether to show edge properties in labels
            
        Returns:
            Base64-encoded PNG image data, or empty string if failed
        """
        if not self.is_available():
            print(f"❌ NetworkX backend not available: {getattr(self, '_import_error', 'Unknown error')}")
            return ""
        
        try:
            import base64
            import io
            
            # Prepare graph data
            G, node_colors, node_labels, edge_labels, edges_added = self._prepare_graph(
                nodes, relationships, show_node_properties, show_edge_properties
            )
            
            if len(G.nodes) == 0:
                print("❌ No valid nodes to visualize")
                return ""
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Setup plot
            self._configure_plot(nodes, show_node_properties, show_edge_properties)
            
            # Calculate layout positions  
            pos = self._get_layout(G, layout)
            
            # Get drawing parameters
            params = self._get_drawing_parameters(show_node_properties, show_edge_properties)
            
            # Draw graph elements
            self._draw_graph_elements(G, pos, node_colors, node_labels, edge_labels, params)
            
            # Add legend
            legend_elements = self._create_legend(nodes)
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right')
            
            ax.axis('off')
            fig.tight_layout()
            
            # Save to bytes buffer
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)  # Clean up
            
            # Convert to base64
            img_buffer.seek(0)
            base64_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            img_buffer.close()
            
            return f"data:image/png;base64,{base64_data}"  # NetworkX uses matplotlib, so PNG only
            
        except Exception as e:
            print(f"❌ Error generating NetworkX base64 image: {e}")
            return ""
    
    # ========== Core Visualization Logic ==========
    
    def _create_visualization(self, nodes: List[GraphNode], relationships: List[GraphRelationship],
                            layout: str, show_node_properties: bool, show_edge_properties: bool,
                            output_path: str = None) -> str:
        """Common visualization logic for both file output and interactive display.
        
        Args:
            output_path: If None, display interactively; if string, save to file
            
        Returns:
            Path to saved file, or empty string if failed/interactive
        """
        if not self.is_available():
            error_msg = f"❌ NetworkX backend not available: {getattr(self, '_import_error', 'Unknown error')}"
            print(error_msg)
            return "" if output_path else None
        
        try:
            # Prepare graph data
            G, node_colors, node_labels, edge_labels, edges_added = self._prepare_graph(
                nodes, relationships, show_node_properties, show_edge_properties
            )
            
            if len(G.nodes) == 0:
                print("❌ No valid nodes to visualize")
                return "" if output_path else None
            
            # Setup plot
            self._configure_plot(nodes, show_node_properties, show_edge_properties)
            
            # Calculate layout positions  
            pos = self._get_layout(G, layout)
            
            # Get drawing parameters
            params = self._get_drawing_parameters(show_node_properties, show_edge_properties)
            
            # Draw graph elements
            self._draw_graph_elements(G, pos, node_colors, node_labels, edge_labels, params)
            
            # Add legend
            legend_elements = self._create_legend(nodes)
            if legend_elements:
                plt.legend(handles=legend_elements, loc='upper right')
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save to file or display interactively
            if output_path:
                # Save to file
                self._finalize_plot(nodes, output_path)
                print(f"NetworkX graph image saved to: {output_path}")
                return output_path
            else:
                # Display interactively
                plt.show()
                print(f"Displayed interactive graph with {len(G.nodes)} nodes and {edges_added} edges")
                return ""
                
        except Exception as e:
            error_msg = f"❌ Error creating NetworkX visualization: {e}"
            print(error_msg)
            if traceback:
                print(f"❌ Full traceback: {traceback.format_exc()}")
            return "" if output_path else None
    
    # ========== Graph Preparation ==========
    
    def _prepare_graph(self, nodes: List[GraphNode], relationships: List[GraphRelationship],
                      show_node_properties: bool, show_edge_properties: bool) -> Tuple:
        """Prepare NetworkX graph with nodes and edges.
        
        Returns:
            Tuple of (graph, node_colors, node_labels, edge_labels, edges_added)
        """
        
        G = nx.Graph()
        node_colors = []
        node_labels = {}
        
        # Add nodes
        for node in nodes:
            G.add_node(node.id)
            label = self.create_node_label(node, show_node_properties, max_label_length=60)
            label = label.replace('\\n', '\n')  # NetworkX uses regular newlines
            node_labels[node.id] = label
            node_colors.append(self.get_node_color(node.type))
        
        # Add edges
        edge_labels = {}
        edges_added = 0
        
        for rel in relationships:
            if rel.source_id in G.nodes and rel.target_id in G.nodes:
                G.add_edge(rel.source_id, rel.target_id)
                edge_label = self.create_edge_label(rel, show_edge_properties, max_label_length=40)
                edge_label = edge_label.replace('\\n', '\n')
                edge_labels[(rel.source_id, rel.target_id)] = edge_label
                edges_added += 1
            else:
                print(f"⚠️ Skipping edge {rel.source_id} -> {rel.target_id} (nodes not found)")
        
        print(f"🔗 Added {edges_added} edges to graph visualization")
        return G, node_colors, node_labels, edge_labels, edges_added
    
    # ========== Layout & Styling ==========
    
    def _get_layout(self, graph, layout: str):
        """Get the specified layout for NetworkX graph."""
        
        layout_map = {
            "spring": lambda g: nx.spring_layout(g, k=self.LAYOUT_SPRING_K, iterations=self.LAYOUT_ITERATIONS),
            "circular": nx.circular_layout,
            "shell": nx.shell_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "random": nx.random_layout
        }
        
        layout_func = layout_map.get(layout, layout_map["spring"])
        return layout_func(graph)
    
    def _create_legend(self, nodes: List[GraphNode]):
        """Create legend for NetworkX visualization."""
        
        detected_types = {node.type for node in nodes}
        return [mpatches.Patch(color=self.get_node_color(node_type), label=node_type) 
                for node_type in sorted(detected_types)]
    
    def _configure_plot(self, nodes: List[GraphNode], show_node_properties: bool, 
                       show_edge_properties: bool):
        """Configure matplotlib plot settings."""
        
        # Configure plot size
        show_props = show_node_properties or show_edge_properties
        figure_size = self.LARGE_FIGURE_SIZE if show_props else self.DEFAULT_FIGURE_SIZE
        plt.figure(figsize=figure_size)
        
        # Set title
        title = self.get_diagram_title(nodes, show_props)
        plt.title(title, fontsize=self.TITLE_FONT_SIZE, fontweight='bold')
        
        return show_props
    
    # ========== Drawing Configuration ==========
    
    def _get_drawing_parameters(self, show_node_properties: bool, show_edge_properties: bool) -> Dict:
        """Get drawing parameters based on property display settings."""
        return {
            'node_size': self.DEFAULT_NODE_SIZE * 3 if show_node_properties else self.DEFAULT_NODE_SIZE * 2,
            'font_size': self.DEFAULT_FONT_SIZE if show_node_properties else self.DEFAULT_FONT_SIZE + 2,
            'edge_font_size': self.DEFAULT_FONT_SIZE - 2 if show_edge_properties else self.DEFAULT_FONT_SIZE
        }
    
    def _draw_graph_elements(self, G, pos, node_colors, node_labels, edge_labels, params):
        """Draw all graph elements using NetworkX."""
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=params['node_size'], alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                             width=2, alpha=0.6, arrows=True, arrowsize=20)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, node_labels, font_size=params['font_size'], 
                              font_weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                                                           facecolor="white", alpha=0.8))
        
        # Draw edge labels if present
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=params['edge_font_size'],
                                       bbox=dict(boxstyle="round,pad=0.2", 
                                                facecolor="yellow", alpha=0.7))
    
    # ========== Plot Finalization ==========
    
    def _finalize_plot(self, nodes: List[GraphNode], output_path: str):
        """Finalize and save the plot."""
        
        # Add legend
        legend_elements = self._create_legend(nodes)
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right')
        
        # Finalize plot
        plt.axis('off')
        plt.tight_layout()
        
        # Save image
        plt.savefig(output_path, dpi=self.HIGH_DPI, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
    

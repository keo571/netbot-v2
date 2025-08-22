"""
Graphviz-based graph visualization.

This module provides high-quality graph visualization using Graphviz.
It produces publication-ready images with precise control over node shapes,
edge routing, and overall aesthetics.
"""

from typing import List, Dict
from models.graph_models import GraphNode, GraphRelationship
from .base import BaseVisualizer, PropertySummaryMixin

# Lazy import of optional dependencies
try:
    import graphviz
    import traceback
    # Test if graphviz executable is available
    test_dot = graphviz.Digraph()
    test_dot.node('test', 'test')
    test_dot.pipe(format='svg')  # Try to render
    _DEPENDENCIES_AVAILABLE = True
except Exception as e:
    graphviz = None
    traceback = None
    _DEPENDENCIES_AVAILABLE = False
    _IMPORT_ERROR = str(e)


class GraphvizVisualizer(BaseVisualizer, PropertySummaryMixin):
    """Graph visualizer using Graphviz backend.
    
    This visualizer provides high-quality, publication-ready graph visualization
    using Graphviz. It supports multiple output formats, precise layout control,
    and professional-grade styling options.
    """
    
    # Configuration constants
    DEFAULT_SIZE = '12,8'
    LARGE_SIZE = '16,12'  # For plots with properties
    DEFAULT_DPI = '300'
    DEFAULT_BGCOLOR = 'white'
    DEFAULT_RANKDIR = 'TB'  # Top to Bottom
    
    # Node styling constants
    DEFAULT_NODE_SHAPE = 'ellipse'
    BOX_NODE_SHAPE = 'box'
    DEFAULT_NODE_STYLE = 'filled'
    ROUNDED_STYLE = 'rounded,filled'
    DEFAULT_FONTNAME = 'Arial'
    DEFAULT_FONTSIZE = '10'
    LARGE_FONTSIZE = '12'
    
    # Edge styling constants
    DEFAULT_EDGE_COLOR = 'black'
    DEFAULT_EDGE_STYLE = 'solid'
    DEFAULT_ARROW_TYPE = 'normal'
    
    def __init__(self):
        """Initialize Graphviz visualizer."""
        super().__init__()
        self._dependencies_available = _DEPENDENCIES_AVAILABLE
        if not _DEPENDENCIES_AVAILABLE:
            self._import_error = _IMPORT_ERROR
    
    def is_available(self) -> bool:
        """Check if Graphviz backend is available."""
        return self._dependencies_available
    
    # ========== Public Interface ==========
    
    def generate_image(self, nodes: List[GraphNode], relationships: List[GraphRelationship], 
                      output_path: str = "subgraph", format: str = "png",
                      layout: str = "dot", show_node_properties: bool = True, 
                      show_edge_properties: bool = True) -> str:
        """Generate a graph image using Graphviz.
        
        Args:
            nodes: List of nodes to visualize
            relationships: List of relationships to visualize  
            output_path: Base path for output file (extension added automatically)
            format: Output format ('png', 'svg', 'pdf', 'ps')
            layout: Graphviz layout engine ('dot', 'neato', 'fdp', 'circo', 'twopi')
            show_node_properties: Whether to show node properties in labels
            show_edge_properties: Whether to show edge properties in labels
            
        Returns:
            Path to generated image file, or empty string if failed
        """
        if not self.is_available():
            print(f"❌ Graphviz backend not available: {getattr(self, '_import_error', 'Unknown error')}")
            return ""
        
        try:
            # Create Graphviz graph
            dot = self._create_graph(nodes, relationships, layout, 
                                   show_node_properties, show_edge_properties)
            
            # Generate output file
            output_file = f"{output_path}.{format}"
            dot.render(output_path, format=format, cleanup=True)
            
            print(f"✅ Graphviz graph saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error generating Graphviz graph: {e}")
            if traceback:
                print(f"❌ Full traceback: {traceback.format_exc()}")
            return ""
    
    def generate_dot_source(self, nodes: List[GraphNode], relationships: List[GraphRelationship],
                           layout: str = "dot", show_node_properties: bool = True,
                           show_edge_properties: bool = True) -> str:
        """Generate DOT source code for the graph.
        
        Args:
            nodes: List of nodes to visualize
            relationships: List of relationships to visualize
            layout_engine: Graphviz layout engine
            show_node_properties: Whether to show node properties
            show_edge_properties: Whether to show edge properties
            
        Returns:
            DOT source code as string
        """
        if not self.is_available():
            return ""
        
        try:
            dot = self._create_graph(nodes, relationships, layout,
                                   show_node_properties, show_edge_properties)
            return dot.source
        except Exception as e:
            print(f"❌ Error generating DOT source: {e}")
            return ""
    
    
    # ========== Core Graph Creation ==========
    
    def _create_graph(self, nodes: List[GraphNode], relationships: List[GraphRelationship],
                     layout: str, show_node_properties: bool, show_edge_properties: bool):
        """Create a Graphviz graph with all nodes and edges."""
        dot = graphviz.Digraph(comment='Knowledge Graph')
        
        # Configure graph appearance
        self._configure_graph_appearance(dot, show_node_properties, show_edge_properties)
        
        # Set layout engine
        self._set_layout_engine(dot, layout)
        
        # Add nodes and edges
        self._add_nodes(dot, nodes, show_node_properties)
        self._add_edges(dot, nodes, relationships, show_edge_properties)
        
        return dot
    
    # ========== Graph Configuration ==========
    
    def _configure_graph_appearance(self, dot, show_node_properties: bool, show_edge_properties: bool):
        """Configure Graphviz graph appearance settings."""
        size = self.LARGE_SIZE if show_node_properties or show_edge_properties else self.DEFAULT_SIZE
        dot.attr(rankdir=self.DEFAULT_RANKDIR, size=size, dpi=self.DEFAULT_DPI, bgcolor=self.DEFAULT_BGCOLOR)
        
        # Node appearance
        if show_node_properties:
            dot.attr('node', shape=self.BOX_NODE_SHAPE, style=self.ROUNDED_STYLE, 
                    fontname=self.DEFAULT_FONTNAME, fontsize='9', margin='0.3,0.2')
        else:
            dot.attr('node', shape=self.BOX_NODE_SHAPE, style=self.ROUNDED_STYLE, 
                    fontname=self.DEFAULT_FONTNAME, fontsize='11', margin='0.1,0.1')
        
        # Edge appearance
        edge_fontsize = '8' if show_edge_properties else self.DEFAULT_FONTSIZE
        dot.attr('edge', fontname=self.DEFAULT_FONTNAME, fontsize=edge_fontsize, color='gray70')
    
    def _set_layout_engine(self, dot, layout: str = 'dot'):
        """Set the Graphviz layout engine."""
        valid_engines = ['dot', 'neato', 'fdp', 'circo', 'twopi', 'sfdp', 'patchwork', 'osage']
        if layout in valid_engines:
            dot.engine = layout
        else:
            print(f"⚠️ Unknown layout engine '{layout}', using 'dot'")
            dot.engine = 'dot'
    
    # ========== Node Styling ==========
    
    def _get_node_shape(self, node_type: str) -> str:
        """Get appropriate node shape based on node type."""
        shape_map = {
            # Flowchart shapes
            'Decision': 'diamond',
            'Start': 'ellipse',
            'End': 'ellipse',
            'Terminal': 'ellipse',
            'Document': 'note',
            'Data': 'parallelogram',
            'Manual': 'trapezium',
            'Subprocess': 'box3d',
            
            # Network shapes
            'Database': 'cylinder',
            'Storage': 'folder',
            'Server': 'box',
            'LoadBalancer': 'hexagon',
            
            # Default
            'default': 'box'
        }
        
        return shape_map.get(node_type, shape_map['default'])
    
    # ========== Edge Styling ==========
    
    def _get_edge_style(self, relationship_type: str) -> Dict[str, str]:
        """Get edge styling based on relationship type."""
        style_map = {
            # Network relationships
            'CONNECTS_TO': {'style': 'solid', 'arrowhead': 'normal'},
            'ROUTES_TO': {'style': 'solid', 'arrowhead': 'normal', 'color': 'blue'},
            'MANAGES': {'style': 'dashed', 'arrowhead': 'diamond'},
            'DEPENDS_ON': {'style': 'dotted', 'arrowhead': 'normal'},
            
            # Flowchart relationships
            'FLOWS_TO': {'style': 'solid', 'arrowhead': 'normal'},
            'BRANCHES_TO': {'style': 'solid', 'arrowhead': 'normal'},
            'RETURNS_TO': {'style': 'dashed', 'arrowhead': 'normal'},
            'CALLS': {'style': 'bold', 'arrowhead': 'normal'},
            
            # Default
            'default': {'style': 'solid', 'arrowhead': 'normal'}
        }
        
        return style_map.get(relationship_type, style_map['default'])
    
    # ========== Graph Construction ==========
    
    def _add_nodes(self, dot, nodes: List[GraphNode], show_properties: bool):
        """Add nodes to Graphviz digraph with appropriate styling."""
        for node in nodes:
            label = self.create_node_label(node, show_properties, max_label_length=120)
            color = self.get_node_color(node.type)
            shape = self._get_node_shape(node.type)
            
            # Configure node with shape and color
            dot.node(node.id, label, fillcolor=color, shape=shape)
    
    def _add_edges(self, dot, nodes: List[GraphNode], relationships: List[GraphRelationship], 
                  show_properties: bool):
        """Add edges to Graphviz digraph."""
        node_ids = {node.id for node in nodes}
        
        for rel in relationships:
            if rel.source_id in node_ids and rel.target_id in node_ids:
                edge_label = self.create_edge_label(rel, show_properties, max_label_length=100)
                
                # Configure edge style based on relationship type
                edge_attrs = self._get_edge_style(rel.type)
                dot.edge(rel.source_id, rel.target_id, label=edge_label, **edge_attrs)


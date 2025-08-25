"""
Visualization service for GraphRAG results.

Provides graph visualization using various backends (GraphViz, NetworkX).
"""

import base64
from typing import List, Dict, Any, Optional
from io import BytesIO

from ...shared import get_logger, GraphNode, GraphRelationship
from .models import VisualizationRequest, VisualizationResult


class VisualizationService:
    """
    Service for creating graph visualizations.
    
    Supports multiple backends and output formats with customizable styling.
    """
    
    def __init__(self):
        """Initialize the visualization service."""
        self.logger = get_logger(__name__)
    
    def create_visualization(self,
                           nodes: List[GraphNode],
                           relationships: List[GraphRelationship],
                           request: VisualizationRequest) -> VisualizationResult:
        """
        Create a graph visualization.
        
        Args:
            nodes: Nodes to visualize
            relationships: Relationships to visualize
            request: Visualization parameters
            
        Returns:
            Visualization result with image data
        """
        try:
            if request.backend.value == "graphviz":
                return self._create_graphviz_visualization(nodes, relationships, request)
            elif request.backend.value == "networkx":
                return self._create_networkx_visualization(nodes, relationships, request)
            else:
                return VisualizationResult(
                    visualization_data="",
                    format=request.format,
                    success=False,
                    error_message=f"Unsupported backend: {request.backend}"
                )
        
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
            return VisualizationResult(
                visualization_data="",
                format=request.format,
                success=False,
                error_message=str(e)
            )
    
    def _create_graphviz_visualization(self,
                                     nodes: List[GraphNode],
                                     relationships: List[GraphRelationship],
                                     request: VisualizationRequest) -> VisualizationResult:
        """Create visualization using GraphViz."""
        try:
            import graphviz
            
            # Create a new directed graph
            dot = graphviz.Digraph(comment='Graph Visualization')
            dot.attr(rankdir='TB', size=f'{request.width/100},{request.height/100}')
            
            # Set layout engine
            layout = request.layout or 'dot'
            dot.engine = layout
            
            # Add nodes
            node_colors = self._get_node_colors(nodes, request.node_color_by)
            
            for node in nodes:
                label = self._create_node_label(node, request.show_node_properties)
                color = node_colors.get(node.id, '#lightblue')
                
                dot.node(
                    node.id,
                    label=label,
                    style='filled',
                    fillcolor=color,
                    fontsize=str(request.font_size)
                )
            
            # Add edges/relationships
            edge_colors = self._get_edge_colors(relationships, request.edge_color_by)
            
            for rel in relationships:
                label = self._create_edge_label(rel, request.show_edge_properties)
                color = edge_colors.get(rel.id, 'black')
                
                dot.edge(
                    rel.source_id,
                    rel.target_id,
                    label=label,
                    color=color,
                    fontsize=str(max(8, request.font_size - 2))
                )
            
            # Generate the visualization
            if request.format.lower() == 'svg':
                svg_data = dot.pipe(format='svg', encoding='utf-8')
                base64_data = base64.b64encode(svg_data.encode()).decode('utf-8')
                return VisualizationResult(
                    visualization_data=f"data:image/svg+xml;base64,{base64_data}",
                    format='svg',
                    metadata={'backend': 'graphviz', 'layout': layout}
                )
            else:
                # PNG format
                png_data = dot.pipe(format='png')
                base64_data = base64.b64encode(png_data).decode('utf-8')
                return VisualizationResult(
                    visualization_data=f"data:image/png;base64,{base64_data}",
                    format='png',
                    metadata={'backend': 'graphviz', 'layout': layout}
                )
        
        except ImportError:
            return VisualizationResult(
                visualization_data="",
                format=request.format,
                success=False,
                error_message="GraphViz not installed. Please install with: pip install graphviz"
            )
        except Exception as e:
            self.logger.error(f"GraphViz visualization failed: {e}")
            return VisualizationResult(
                visualization_data="",
                format=request.format,
                success=False,
                error_message=f"GraphViz error: {str(e)}"
            )
    
    def _create_networkx_visualization(self,
                                     nodes: List[GraphNode],
                                     relationships: List[GraphRelationship],
                                     request: VisualizationRequest) -> VisualizationResult:
        """Create visualization using NetworkX and Matplotlib."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes
            for node in nodes:
                G.add_node(
                    node.id,
                    label=node.label,
                    type=node.type,
                    **node.properties
                )
            
            # Add edges
            for rel in relationships:
                G.add_edge(
                    rel.source_id,
                    rel.target_id,
                    type=rel.type,
                    **rel.properties
                )
            
            # Create layout
            layout = request.layout or 'spring'
            if layout == 'spring':
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'shell':
                pos = nx.shell_layout(G)
            else:
                pos = nx.spring_layout(G)  # Default fallback
            
            # Create figure
            fig, ax = plt.subplots(figsize=(request.width/100, request.height/100))
            
            # Get colors
            node_colors = self._get_networkx_node_colors(nodes, request.node_color_by)
            edge_colors = self._get_networkx_edge_colors(relationships, request.edge_color_by)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_color=node_colors,
                node_size=request.node_size,
                alpha=0.7,
                ax=ax
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                edge_color=edge_colors,
                alpha=0.6,
                arrows=True,
                arrowsize=20,
                ax=ax
            )
            
            # Add labels
            if request.show_node_properties:
                labels = {node.id: self._create_node_label(node, True) for node in nodes}
            else:
                labels = {node.id: node.label for node in nodes}
            
            nx.draw_networkx_labels(
                G, pos,
                labels=labels,
                font_size=request.font_size,
                ax=ax
            )
            
            # Add edge labels if requested
            if request.show_edge_properties:
                edge_labels = {
                    (rel.source_id, rel.target_id): self._create_edge_label(rel, True)
                    for rel in relationships
                }
                nx.draw_networkx_edge_labels(
                    G, pos,
                    edge_labels=edge_labels,
                    font_size=max(8, request.font_size - 2),
                    ax=ax
                )
            
            ax.set_title("Graph Visualization", fontsize=request.font_size + 2)
            plt.axis('off')
            
            # Convert to base64
            buffer = BytesIO()
            if request.format.lower() == 'svg':
                plt.savefig(buffer, format='svg', bbox_inches='tight')
                buffer.seek(0)
                svg_data = buffer.getvalue()
                base64_data = base64.b64encode(svg_data).decode('utf-8')
                result_data = f"data:image/svg+xml;base64,{base64_data}"
            else:
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                png_data = buffer.getvalue()
                base64_data = base64.b64encode(png_data).decode('utf-8')
                result_data = f"data:image/png;base64,{base64_data}"
            
            plt.close(fig)
            
            return VisualizationResult(
                visualization_data=result_data,
                format=request.format,
                metadata={'backend': 'networkx', 'layout': layout}
            )
        
        except ImportError:
            return VisualizationResult(
                visualization_data="",
                format=request.format,
                success=False,
                error_message="NetworkX or Matplotlib not installed. Please install with: pip install networkx matplotlib"
            )
        except Exception as e:
            self.logger.error(f"NetworkX visualization failed: {e}")
            return VisualizationResult(
                visualization_data="",
                format=request.format,
                success=False,
                error_message=f"NetworkX error: {str(e)}"
            )
    
    def _create_node_label(self, node: GraphNode, show_properties: bool) -> str:
        """Create node label with optional properties."""
        label = f"{node.label}\\n({node.type})"
        
        if show_properties and node.properties:
            # Add a few key properties
            prop_lines = []
            excluded = {'id', 'diagram_id', 'embedding'}
            
            for key, value in list(node.properties.items())[:3]:  # Limit to 3 props
                if key not in excluded and value:
                    prop_lines.append(f"{key}: {str(value)[:20]}")
            
            if prop_lines:
                label += "\\n" + "\\n".join(prop_lines)
        
        return label
    
    def _create_edge_label(self, relationship: GraphRelationship, show_properties: bool) -> str:
        """Create edge label with optional properties."""
        label = relationship.type
        
        if show_properties and relationship.properties:
            # Add key properties
            prop_parts = []
            excluded = {'id', 'diagram_id'}
            
            for key, value in list(relationship.properties.items())[:2]:  # Limit to 2 props
                if key not in excluded and value:
                    prop_parts.append(f"{key}: {str(value)[:15]}")
            
            if prop_parts:
                label += "\\n" + "\\n".join(prop_parts)
        
        return label
    
    def _get_node_colors(self, nodes: List[GraphNode], color_by: Optional[str]) -> Dict[str, str]:
        """Get color mapping for nodes."""
        colors = {}
        
        if not color_by or color_by == 'type':
            # Color by node type
            type_colors = {
                'server': '#ff6b6b',
                'database': '#4ecdc4', 
                'loadbalancer': '#45b7d1',
                'router': '#f9ca24',
                'switch': '#6c5ce7',
                'firewall': '#fd79a8',
                'process': '#00b894',
                'decision': '#fdcb6e',
                'start': '#55a3ff',
                'end': '#ff6b9d'
            }
            
            for node in nodes:
                node_type = node.type.lower()
                colors[node.id] = type_colors.get(node_type, '#lightgray')
        
        else:
            # Default coloring
            for node in nodes:
                colors[node.id] = '#lightblue'
        
        return colors
    
    def _get_edge_colors(self, relationships: List[GraphRelationship], color_by: Optional[str]) -> Dict[str, str]:
        """Get color mapping for edges."""
        colors = {}
        
        if not color_by or color_by == 'type':
            # Color by relationship type
            type_colors = {
                'connects_to': '#2d3436',
                'flows_to': '#0984e3',
                'contains': '#6c5ce7',
                'depends_on': '#e17055',
                'manages': '#00b894'
            }
            
            for rel in relationships:
                rel_type = rel.type.lower()
                colors[rel.id] = type_colors.get(rel_type, 'black')
        
        else:
            # Default coloring
            for rel in relationships:
                colors[rel.id] = 'black'
        
        return colors
    
    def _get_networkx_node_colors(self, nodes: List[GraphNode], color_by: Optional[str]) -> List[str]:
        """Get color list for NetworkX nodes."""
        node_colors = self._get_node_colors(nodes, color_by)
        return [node_colors[node.id] for node in nodes]
    
    def _get_networkx_edge_colors(self, relationships: List[GraphRelationship], color_by: Optional[str]) -> List[str]:
        """Get color list for NetworkX edges."""
        edge_colors = self._get_edge_colors(relationships, color_by)
        return [edge_colors[rel.id] for rel in relationships]
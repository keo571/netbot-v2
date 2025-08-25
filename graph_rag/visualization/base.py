"""
Base classes and shared utilities for graph visualization.

This module contains common constants, configuration, and utility methods
shared across different visualization backends (NetworkX, Graphviz, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set
from models.graph_models import GraphNode, GraphRelationship


class VisualizationConfig:
    """Shared configuration constants for all visualization backends."""
    
    # Color schemes for different node types
    NETWORK_COLORS = {
        'LoadBalancer': '#FF6B6B',
        'Server': '#4ECDC4', 
        'Database': '#45B7D1',
        'Network': '#96CEB4',
        'Storage': '#9B59B6',
        'Router': '#E67E22',
        'Switch': '#F39C12',
        'Firewall': '#E74C3C',
        'Gateway': '#1ABC9C',
    }
    
    FLOWCHART_COLORS = {
        'Process': '#FFEAA7',
        'Decision': '#FFD93D',
        'Start': '#6C5CE7',
        'End': '#A29BFE',
        'Terminal': '#A29BFE',
        'Document': '#74B9FF',
        'Data': '#81ECEC',
        'Manual': '#FDCB6E',
        'Connector': '#DDA0DD',
        'Subprocess': '#FD79A8',
        'Preparation': '#FDCB6E',
        'Input': '#00B894',
        'Output': '#00CEC9',
    }
    
    GENERIC_COLORS = {
        'Node': '#CCCCCC',
        'Component': '#BDC3C7',
        'Entity': '#95A5A6',
    }
    
    # Combined color scheme
    ALL_COLORS = {**NETWORK_COLORS, **FLOWCHART_COLORS, **GENERIC_COLORS}
    
    # Default settings
    DEFAULT_COLOR = '#CCCCCC'
    DEFAULT_EXCLUDED_PROPERTIES = {'id', 'diagram_id', 'embedding'}
    
    # Node type classifications
    NETWORK_TYPES = {'LoadBalancer', 'Server', 'Database', 'Network', 'Router', 'Switch', 'Firewall', 'Gateway', 'Storage'}
    FLOWCHART_TYPES = {'Decision', 'Start', 'End', 'Terminal', 'Document', 'Data', 'Manual', 'Subprocess', 'Preparation', 'Input', 'Output'}
    
    # Diagram type titles
    DIAGRAM_TITLES = {
        'network': 'Network Diagram',
        'flowchart': 'Flowchart',
        'generic': 'Knowledge Graph'
    }


class BaseVisualizer(ABC):
    """Abstract base class for graph visualizers."""
    
    def __init__(self):
        """Initialize base visualizer with shared configuration."""
        self.config = VisualizationConfig()
        self.color_scheme = self.config.ALL_COLORS
        self.default_excluded_properties = self.config.DEFAULT_EXCLUDED_PROPERTIES
    
    # ========== Shared Utility Methods ==========
    
    def get_node_color(self, node_type: str) -> str:
        """Get color for a node type, with fallback to default."""
        return self.color_scheme.get(node_type, self.config.DEFAULT_COLOR)
    
    def detect_diagram_type(self, nodes: List[GraphNode]) -> str:
        """Detect diagram type based on node types present.
        
        Returns:
            'network', 'flowchart', or 'generic' based on node types
        """
        node_types = {node.type for node in nodes}
        
        network_score = len(node_types & self.config.NETWORK_TYPES)
        flowchart_score = len(node_types & self.config.FLOWCHART_TYPES)
        
        if network_score > flowchart_score:
            return "network"
        elif flowchart_score > 0:
            return "flowchart"
        else:
            return "generic"
    
    def get_diagram_title(self, nodes: List[GraphNode], show_properties: bool = False) -> str:
        """Generate appropriate title based on diagram type and options."""
        diagram_type = self.detect_diagram_type(nodes)
        title = self.config.DIAGRAM_TITLES.get(diagram_type, 'Knowledge Graph')
        
        if show_properties:
            title += " (with Properties)"
        
        return title
    
    def format_properties_text(self, properties: Dict, excluded_props: Optional[Set[str]] = None, 
                              max_value_length: int = 50) -> str:
        """Format properties dictionary into readable text for labels.
        
        Args:
            properties: Dictionary of properties to format
            excluded_props: Set of property keys to exclude
            max_value_length: Maximum length for property values before truncation
        """
        if not properties:
            return ""
        
        exclude = excluded_props or self.default_excluded_properties
        filtered_props = {k: v for k, v in properties.items() 
                         if k not in exclude and v is not None and str(v).strip()}
        
        if not filtered_props:
            return ""
        
        prop_lines = []
        for key, value in filtered_props.items():
            value_str = str(value)
            if len(value_str) > max_value_length:
                value_str = value_str[:max_value_length-3] + "..."
            prop_lines.append(f"{key}: {value_str}")
        
        return "\\n".join(prop_lines)  # Use escaped newline for Graphviz compatibility
    
    def create_base_label(self, item: GraphNode) -> str:
        """Create base label for a node."""
        return item.label if item.label else f"{item.type}_{item.id[:8]}"
    
    def truncate_label(self, label: str, max_length: int) -> str:
        """Truncate label if it exceeds maximum length."""
        if len(label) > max_length:
            return label[:max_length-3] + "..."
        return label
    
    def create_node_label(self, node: GraphNode, show_properties: bool = True, 
                         max_label_length: int = 100) -> str:
        """Create a comprehensive node label with optional properties."""
        base_label = self.create_base_label(node)
        
        if not show_properties:
            return base_label
        
        props_text = self.format_properties_text(node.properties)
        if props_text:
            full_label = f"{base_label}\\n---\\n{props_text}"
        else:
            full_label = base_label
        
        return self.truncate_label(full_label, max_label_length)
    
    def create_edge_label(self, relationship: GraphRelationship, show_properties: bool = True,
                         max_label_length: int = 80) -> str:
        """Create a comprehensive edge label with optional properties."""
        base_label = relationship.type
        
        if not show_properties:
            return base_label
        
        props_text = self.format_properties_text(relationship.properties)
        if props_text:
            full_label = f"{base_label}\\n{props_text}"
        else:
            full_label = base_label
        
        return self.truncate_label(full_label, max_label_length)
    
    # ========== Abstract Methods ==========
    
    @abstractmethod
    def generate_image(self, nodes: List[GraphNode], relationships: List[GraphRelationship], 
                      output_path: str, **kwargs) -> str:
        """Generate graph visualization image.
        
        Args:
            nodes: List of nodes to visualize
            relationships: List of relationships to visualize
            output_path: Path for output image file
            **kwargs: Backend-specific options
            
        Returns:
            Path to generated image file, or empty string if failed
        """
        raise NotImplementedError("Subclasses must implement generate_image()")
    
    @abstractmethod
    def generate_image_base64(self, nodes: List[GraphNode], relationships: List[GraphRelationship], 
                             **kwargs) -> str:
        """Generate graph visualization as base64 string without saving file.
        
        Args:
            nodes: List of nodes to visualize
            relationships: List of relationships to visualize
            **kwargs: Backend-specific options
            
        Returns:
            Base64-encoded PNG image data, or empty string if failed
        """
        raise NotImplementedError("Subclasses must implement generate_image_base64()")
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this visualization backend is available.
        
        Returns:
            True if backend dependencies are installed and working
        """
        raise NotImplementedError("Subclasses must implement is_available()")


class PropertySummaryMixin:
    """Mixin class providing property summary functionality."""
    
    def format_node_summary(self, node: GraphNode, index: int) -> List[str]:
        """Format a single node for the property summary."""
        lines = []
        lines.append(f"\n{index}. {node.label or node.type} (ID: {node.id[:12]})")
        lines.append(f"   Type: {node.type}")
        
        if node.properties:
            filtered_props = {k: v for k, v in node.properties.items() 
                            if k not in self.default_excluded_properties and v is not None}
            
            if filtered_props:
                lines.append("   Properties:")
                for key, value in filtered_props.items():
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:97] + "..."
                    lines.append(f"     • {key}: {value_str}")
        
        return lines
    
    def format_relationship_summary(self, rel: GraphRelationship, nodes: List[GraphNode], index: int) -> List[str]:
        """Format a single relationship for the property summary."""
        lines = []
        
        # Find node labels
        source_label = next((n.label for n in nodes if n.id == rel.source_id), rel.source_id[:12])
        target_label = next((n.label for n in nodes if n.id == rel.target_id), rel.target_id[:12])
        
        lines.append(f"\n{index}. {source_label} → {target_label}")
        lines.append(f"   Type: {rel.type}")
        
        if rel.properties:
            filtered_props = {k: v for k, v in rel.properties.items() 
                            if k not in self.default_excluded_properties and v is not None}
            
            if filtered_props:
                lines.append("   Properties:")
                for key, value in filtered_props.items():
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:97] + "..."
                    lines.append(f"     • {key}: {value_str}")
        
        return lines
    
    def create_property_summary(self, nodes: List[GraphNode], relationships: List[GraphRelationship]) -> str:
        """Create a comprehensive text summary of all properties in the graph.
        
        Args:
            nodes: List of nodes to summarize
            relationships: List of relationships to summarize
            
        Returns:
            Formatted string containing complete property summary
        """
        summary = []
        summary.append("=" * 60)
        summary.append("GRAPH PROPERTIES SUMMARY")
        summary.append("=" * 60)
        
        # Node properties summary
        summary.append(f"\n📊 NODES ({len(nodes)} total):")
        summary.append("-" * 30)
        
        for i, node in enumerate(nodes, 1):
            summary.extend(self.format_node_summary(node, i))
        
        # Relationship properties summary
        summary.append(f"\n🔗 RELATIONSHIPS ({len(relationships)} total):")
        summary.append("-" * 30)
        
        for i, rel in enumerate(relationships, 1):
            summary.extend(self.format_relationship_summary(rel, nodes, i))
        
        summary.append("\n" + "=" * 60)
        
        return "\n".join(summary)
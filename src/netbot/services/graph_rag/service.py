"""
Graph RAG Service implementation.

Provides graph-based retrieval-augmented generation using the shared infrastructure.
"""

import time
import re
from typing import Dict, Any, List, Optional, Tuple

from ...shared import (
    get_logger, get_metrics, get_model_client, get_embedding_client, 
    GraphNode, GraphRelationship, AIError, DatabaseError
)
from .models import (
    SearchRequest, SearchResult, VisualizationRequest, 
    ExplanationRequest, VisualizationResult
)
from .repository import GraphRAGRepository
from .retrieval import TwoPhaseRetriever
# from .visualization import VisualizationService  # Removed - no longer using server-side rendering


class GraphRAGService:
    """
    Service for graph-based retrieval-augmented generation.
    
    Provides semantic search, explanation generation, and visualization
    capabilities using the shared infrastructure.
    """
    
    def __init__(self):
        """Initialize the GraphRAG service."""
        self.logger = get_logger(__name__)
        self.metrics = get_metrics()
        self.model_client = get_model_client()
        self.embedding_client = get_embedding_client()
        
        # Service components
        self.repository = GraphRAGRepository()
        self.retriever = TwoPhaseRetriever(
            embedding_client=self.embedding_client,
            repository=self.repository
        )
        # self.visualization_service = VisualizationService()  # Removed - no longer using server-side rendering
        
        self.logger.info("GraphRAG Service initialized")
    
    def search(self, request: SearchRequest) -> SearchResult:
        """
        Perform graph search with natural language query.
        
        Args:
            request: Search request with parameters
            
        Returns:
            Search results with nodes, relationships, and metadata
        """
        start_time = time.time()
        result = SearchResult(request=request)
        
        try:
            self.logger.info(f"Searching: {request.query} in {request.diagram_id}")
            
            # Perform two-phase retrieval
            retrieval_result = self.retriever.retrieve(
                query=request.query,
                diagram_id=request.diagram_id,
                method=request.method.value,
                top_k=request.top_k,
                min_similarity=request.min_similarity
            )
            
            # Extract results
            result.nodes = retrieval_result.nodes
            result.relationships = retrieval_result.relationships
            result.metadata = retrieval_result.metadata
            result.relevance_score = retrieval_result.relevance_score
            result.confidence_score = retrieval_result.confidence_score
            
            # Generate explanation if requested
            if request.include_explanation and (result.nodes or result.relationships):
                explanation_request = ExplanationRequest(
                    nodes=result.nodes,
                    relationships=result.relationships,
                    original_query=request.query,
                    detailed=request.detailed_explanation
                )
                result.explanation = self.generate_explanation(explanation_request)
            
            # Generate visualization data if requested (JSON only, no server-side rendering)
            if request.include_visualization and (result.nodes or result.relationships):
                viz_data = self._create_visualization_json(result.nodes, result.relationships, request.diagram_id)
                result.visualization_data = viz_data
            
            self.logger.info(
                f"Search completed: {len(result.nodes)} nodes, "
                f"{len(result.relationships)} relationships"
            )
            
            # Record metrics
            self.metrics.record_api_request(
                endpoint="graph_rag_search",
                method="POST",
                duration_seconds=time.time() - start_time,
                status_code=200
            )
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            
            self.logger.error(f"Search failed for {request.diagram_id}: {e}")
            
            # Record error metrics
            self.metrics.record_api_request(
                endpoint="graph_rag_search",
                method="POST",
                duration_seconds=time.time() - start_time,
                status_code=500
            )
            
            if not isinstance(e, (AIError, DatabaseError)):
                raise AIError(f"Graph search failed: {e}")
        
        finally:
            search_time = time.time() - start_time
            result.metadata.search_time_ms = search_time * 1000
        
        return result
    
    def generate_explanation(self, request: ExplanationRequest) -> str:
        """
        Generate natural language explanation of graph results.
        
        Args:
            request: Explanation request with nodes and relationships
            
        Returns:
            Natural language explanation
        """
        try:
            # Detect diagram type if not provided
            diagram_type = request.diagram_type or self._detect_diagram_type(
                request.nodes, request.relationships
            )
            
            # Format nodes and relationships for explanation
            nodes_text = self._format_nodes_for_explanation(request.nodes)
            relationships_text = self._format_relationships_for_explanation(
                request.relationships, request.nodes
            )
            
            # Create explanation prompt
            prompt = self._create_explanation_prompt(
                diagram_type=diagram_type,
                query=request.original_query,
                nodes_text=nodes_text,
                relationships_text=relationships_text,
                detailed=request.detailed
            )
            
            # Generate explanation using model client
            explanation = self.model_client.generate_text(
                prompt=prompt,
                model_name=request.model_name,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                cache_response=True
            )
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return f"Error generating explanation: {str(e)}"
    
    def visualize(self, 
                 nodes: List[GraphNode],
                 relationships: List[GraphRelationship],
                 request: VisualizationRequest) -> VisualizationResult:
        """
        Generate graph visualization data for frontend rendering.
        
        Args:
            nodes: Nodes to visualize
            relationships: Relationships to visualize
            request: Visualization parameters
            
        Returns:
            Visualization result with JSON data
        """
        try:
            viz_data = self._create_visualization_json(
                nodes=nodes,
                relationships=relationships,
                diagram_id=getattr(request, 'diagram_id', 'unknown')
            )
            return VisualizationResult(
                visualization_data=viz_data,
                visualization_type="network_graph",
                success=True
            )
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")
            return VisualizationResult(
                visualization_data={},
                visualization_type="network_graph",
                success=False,
                error_message=str(e)
            )
    
    def search_and_visualize(self, request: SearchRequest) -> SearchResult:
        """
        Perform search and generate visualization in one operation.
        
        Args:
            request: Search request (include_visualization will be set to True)
            
        Returns:
            Search result with visualization data
        """
        # Force visualization generation
        request.include_visualization = True
        return self.search(request)
    
    def invalidate_cache(self, diagram_id: str) -> None:
        """
        Invalidate cached data for a diagram.
        
        Args:
            diagram_id: Diagram identifier
        """
        try:
            # Clear embedding cache
            self.embedding_client.cache.delete(f"matrix_{diagram_id}", namespace='embeddings')
            
            # Clear any other relevant caches
            cache = self.embedding_client.cache
            cache.clear_namespace('search_results')
            
            self.logger.info(f"Cleared cache for diagram: {diagram_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to clear cache for {diagram_id}: {e}")
    
    def get_diagram_statistics(self, diagram_id: str) -> Dict[str, Any]:
        """
        Get statistics for a diagram.
        
        Args:
            diagram_id: Diagram identifier
            
        Returns:
            Diagram statistics
        """
        try:
            return self.repository.get_diagram_stats(diagram_id)
        except Exception as e:
            self.logger.error(f"Failed to get diagram statistics: {e}")
            return {}
    
    def _detect_diagram_type(self, 
                           nodes: List[GraphNode], 
                           relationships: List[GraphRelationship]) -> str:
        """Detect diagram type from node and relationship patterns."""
        network_indicators = 0
        flowchart_indicators = 0
        
        # Network terms patterns
        network_terms = [
            r'\\b(router|switch|firewall|gateway|hub|bridge|access point|ap|server|database|load balancer)\\b',
            r'\\b(eth|gi|fa|se|lo|en|wlan)\\d+',
            r'\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b',  # IP addresses
            r'\\b(vlan|subnet|network|interface|port|connection)\\b'
        ]
        
        # Flowchart terms patterns
        flowchart_terms = [
            r'\\b(start|end|begin|finish|stop|decision|process)\\b',
            r'\\b(yes|no|true|false|approve|reject)\\b',
            r'\\b(if|then|else|while|for|do|check|verify)\\b',
            r'\\?\\s*$',  # Questions ending with ?
            r'\\b(step|phase|stage)\\s*\\d+\\b'
        ]
        
        # Check node types and labels
        node_types = [n.type.lower() for n in nodes]
        node_labels = [n.label.lower() for n in nodes]
        rel_types = [r.type.lower() for r in relationships]
        
        all_text = ' '.join(node_types + node_labels + rel_types)
        
        # Count network indicators
        for pattern in network_terms:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            network_indicators += len(matches)
        
        # Count flowchart indicators  
        for pattern in flowchart_terms:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            flowchart_indicators += len(matches)
        
        # Check node types for specific patterns
        network_node_types = {'server', 'database', 'loadbalancer', 'router', 'switch', 'firewall'}
        flowchart_node_types = {'decision', 'process', 'start', 'end', 'terminal'}
        
        network_indicators += sum(1 for t in node_types if t in network_node_types)
        flowchart_indicators += sum(1 for t in node_types if t in flowchart_node_types)
        
        # Determine type
        if network_indicators > flowchart_indicators * 1.5:
            return 'network'
        elif flowchart_indicators > network_indicators * 1.5:
            return 'flowchart'
        else:
            return 'mixed'
    
    def _format_nodes_for_explanation(self, nodes: List[GraphNode]) -> List[str]:
        """Format nodes for explanation text."""
        nodes_text = []
        excluded_properties = {'id', 'diagram_id', 'embedding'}
        
        for node in nodes:
            props = ", ".join([
                f"{k}: {v}" for k, v in node.properties.items() 
                if k not in excluded_properties
            ])
            label = f"- {node.label} ({node.type})"
            if props:
                label += f" - {props}"
            nodes_text.append(label)
        
        return nodes_text
    
    def _format_relationships_for_explanation(self, 
                                            relationships: List[GraphRelationship],
                                            nodes: List[GraphNode]) -> List[str]:
        """Format relationships for explanation text."""
        rels_text = []
        
        # Create node lookup for labels
        node_labels = {node.id: node.label for node in nodes}
        
        for rel in relationships:
            source_label = node_labels.get(rel.source_id, 'Unknown')
            target_label = node_labels.get(rel.target_id, 'Unknown')
            rels_text.append(f"- {source_label} → {target_label} ({rel.type})")
        
        return rels_text
    
    def _create_explanation_prompt(self,
                                 diagram_type: str,
                                 query: str,
                                 nodes_text: List[str],
                                 relationships_text: List[str],
                                 detailed: bool) -> str:
        """Create explanation prompt based on diagram type and detail level."""
        elements_text = "\\n".join(nodes_text)
        relations_text = "\\n".join(relationships_text)
        
        if detailed:
            return self._create_detailed_prompt(diagram_type, query, elements_text, relations_text)
        else:
            return self._create_simple_prompt(diagram_type, query, elements_text, relations_text)
    
    def _create_detailed_prompt(self, diagram_type: str, query: str, 
                              elements_text: str, relationships_text: str) -> str:
        """Create detailed analysis prompt."""
        prompts = {
            'network': f"""
            Analyze this network diagram subgraph and provide a detailed technical explanation:
            
            Query: "{query}"
            
            Network components:
            {elements_text}
            
            Connections:
            {relationships_text}
            
            Provide a comprehensive analysis covering:
            1. Overview of the network components and their roles
            2. Network architecture patterns identified
            3. Potential security considerations
            4. Performance implications
            5. Best practices recommendations
            """,
            
            'flowchart': f"""
            Analyze this flowchart/process diagram and provide a detailed explanation:
            
            Query: "{query}"
            
            Process elements:
            {elements_text}
            
            Process flow:
            {relationships_text}
            
            Provide a comprehensive analysis covering:
            1. Overview of the process steps and decision points
            2. Workflow patterns and logic identified
            3. Potential bottlenecks or inefficiencies
            4. Exception handling and error paths
            5. Process optimization recommendations
            """,
            
            'mixed': f"""
            Analyze this complex diagram containing multiple element types and provide a detailed explanation:
            
            Query: "{query}"
            
            Diagram elements:
            {elements_text}
            
            Relationships:
            {relationships_text}
            
            Provide a comprehensive analysis covering:
            1. Overview of all components and their roles
            2. System architecture and process patterns identified
            3. Integration points and dependencies
            4. Potential issues or optimization opportunities
            5. Best practices recommendations
            """
        }
        
        return prompts.get(diagram_type, prompts['mixed'])
    
    def _create_simple_prompt(self, diagram_type: str, query: str,
                            elements_text: str, relationships_text: str) -> str:
        """Create simple explanation prompt."""
        prompts = {
            'network': f"""
            Query: "{query}"
            
            Components: {elements_text}
            Connections: {relationships_text}
            
            Provide a brief explanation (3-4 sentences) of what this network shows and how it relates to the query.
            """,
            
            'flowchart': f"""
            Query: "{query}"
            
            Steps: {elements_text}
            Flow: {relationships_text}
            
            Provide a brief explanation (3-4 sentences) of what this process shows and how it relates to the query.
            """,
            
            'mixed': f"""
            Query: "{query}"
            
            Elements: {elements_text}
            Relationships: {relationships_text}
            
            Provide a brief explanation (3-4 sentences) of what this diagram shows and how it relates to the query.
            """
        }
        
        return prompts.get(diagram_type, prompts['mixed'])
    
    def _create_visualization_json(
        self, 
        nodes: List[GraphNode], 
        relationships: List[GraphRelationship],
        diagram_id: str
    ) -> Dict[str, Any]:
        """
        Create JSON visualization data for frontend rendering.
        Fast alternative to server-side image generation.
        
        Args:
            nodes: Graph nodes to visualize
            relationships: Graph relationships to visualize
            diagram_id: Source diagram ID
            
        Returns:
            JSON data for frontend visualization
        """
        # Convert nodes to frontend format
        viz_nodes = []
        for node in nodes:
            viz_nodes.append({
                'id': node.id,
                'label': node.label,
                'type': node.type,
                'properties': node.properties,
                # Add visualization hints for frontend
                'size': min(max(len(str(node.properties)), 20), 100),  # Size based on content
                'color': self._get_node_color(node.type),
                'shape': self._get_node_shape(node.type)
            })
        
        # Convert relationships to frontend format
        viz_relationships = []
        for rel in relationships:
            viz_relationships.append({
                'source': rel.source_id,
                'target': rel.target_id,
                'label': rel.type,
                'type': rel.type,
                'properties': rel.properties,
                # Add visualization hints
                'weight': rel.confidence_score,
                'color': self._get_edge_color(rel.type),
                'style': self._get_edge_style(rel.type)
            })
        
        return {
            'visualization_type': 'network_graph',
            'nodes': viz_nodes,
            'edges': viz_relationships,
            'metadata': {
                'diagram_id': diagram_id,
                'node_count': len(viz_nodes),
                'edge_count': len(viz_relationships),
                'layout_hint': 'force_directed'
            }
        }
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type."""
        color_map = {
            'Server': '#4A90E2',
            'Database': '#50E3C2',
            'LoadBalancer': '#F5A623',
            'Firewall': '#D0021B',
            'Router': '#7ED321',
            'Switch': '#9013FE',
            'Process': '#4A90E2',
            'Decision': '#F5A623',
            'DataStore': '#50E3C2',
            'default': '#B8B8B8'
        }
        return color_map.get(node_type, color_map['default'])
    
    def _get_node_shape(self, node_type: str) -> str:
        """Get shape for node type."""
        shape_map = {
            'Server': 'box',
            'Database': 'cylinder',
            'LoadBalancer': 'diamond',
            'Firewall': 'triangle',
            'Router': 'hexagon',
            'Switch': 'ellipse',
            'Process': 'box',
            'Decision': 'diamond',
            'DataStore': 'cylinder',
            'default': 'circle'
        }
        return shape_map.get(node_type, shape_map['default'])
    
    def _get_edge_color(self, rel_type: str) -> str:
        """Get color for relationship type."""
        color_map = {
            'CONNECTS_TO': '#333333',
            'FLOWS_TO': '#4A90E2',
            'ROUTES_TO': '#7ED321',
            'PROTECTS': '#D0021B',
            'STORES_IN': '#50E3C2',
            'default': '#666666'
        }
        return color_map.get(rel_type, color_map['default'])
    
    def _get_edge_style(self, rel_type: str) -> str:
        """Get style for relationship type."""
        style_map = {
            'PROTECTS': 'dashed',
            'STORES_IN': 'dotted',
            'default': 'solid'
        }
        return style_map.get(rel_type, style_map['default'])
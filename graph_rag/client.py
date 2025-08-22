"""
GraphRAG client interface - Comprehensive and easy to use.

Usage:
    from graph_rag import GraphRAG
    
    # Basic workflow
    rag = GraphRAG()
    results = rag.search("find load balancers", "diagram_001")
    
    # Advanced workflow with visualization and explanation
    results = rag.query_and_visualize(
        "find load balancers", 
        "diagram_001",
        include_explanation=True
    )
"""

import os
import re
import hashlib
from datetime import datetime
import google.generativeai as genai
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Core imports
from .database.connection import Neo4jConnection
from .database.schema_extractor import SchemaExtractor
from .database.query_executor import QueryExecutor
from .search.vector_search import VectorSearch
from .retrieval.cypher_generator import CypherGenerator
from .retrieval.two_phase_retriever import TwoPhaseRetriever
from .visualization import GraphVisualizer


class GraphRAG:
    """
    Simple GraphRAG interface for diagram analysis and knowledge graph search.
    
    Supports multiple diagram types: network diagrams, flowcharts, and knowledge graphs.
    
    Provides a clean API for GraphRAG operations:
    1. Search graphs with natural language
    2. Generate visualizations
    3. Create explanations
    """
    
    # Configuration constants
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_TOP_K = 8
    DEFAULT_SEARCH_METHOD = "auto"
    
    # Model constants
    DEFAULT_GEMINI_MODEL = 'gemini-2.5-flash'
    
    # Property filtering constants
    EXCLUDED_PROPERTIES = {'id', 'diagram_id', 'embedding'}
    
    
    def __init__(self, 
                 neo4j_uri: str = None,
                 neo4j_user: str = None, 
                 neo4j_password: str = None,
                 gemini_api_key: str = None):
        """
        Initialize GraphRAG with database and AI credentials.
        
        Args:
            neo4j_uri: Neo4j database URI (defaults to env NEO4J_URI)
            neo4j_user: Neo4j username (defaults to env NEO4J_USER) 
            neo4j_password: Neo4j password (defaults to env NEO4J_PASSWORD)
            gemini_api_key: Gemini API key (defaults to env GEMINI_API_KEY)
        """
        # Use provided credentials or fall back to environment variables
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        
        # Validate required credentials
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD must be provided via parameter or environment variable")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be provided via parameter or environment variable")
        
        # Initialize components (lazy loading for better performance)
        self._visualizer = None
        
        # Search infrastructure (lazy loading)
        self._connection = None
        self._retriever = None
    
    
    def _ensure_search_components_initialized(self):
        """Ensure search components are initialized"""
        self._get_search_components()
    
    
    def _configure_gemini(self):
        """Configure Gemini API (called once during initialization)"""
        genai.configure(api_key=self.gemini_api_key)
    
    def _ensure_visualizer_initialized(self):
        """Ensure visualizer is initialized"""
        if not self._visualizer:
            self._visualizer = GraphVisualizer()
    
    def _format_nodes_for_explanation(self, nodes):
        """Format nodes for explanation text"""
        nodes_text = []
        for node in nodes:
            props = ", ".join([f"{k}: {v}" for k, v in node.properties.items() 
                             if k not in self.EXCLUDED_PROPERTIES])
            label = f"- {node.label} ({node.type})"
            if props:
                label += f" - {props}"
            nodes_text.append(label)
        return nodes_text
    
    def _format_relationships_for_explanation(self, relationships, nodes):
        """Format relationships for explanation text"""
        rels_text = []
        for rel in relationships:
            source_label = next((n.label for n in nodes 
                               if n.id == rel.source_id), 'Unknown')
            target_label = next((n.label for n in nodes 
                               if n.id == rel.target_id), 'Unknown')
            rels_text.append(f"- {source_label} → {target_label} ({rel.type})")
        return rels_text
    
    def _detect_diagram_type(self, nodes: List[Dict], relationships: List[Dict]) -> str:
        """Detect diagram type from node and relationship patterns"""
        network_indicators = 0
        flowchart_indicators = 0
        
        # Network terms patterns
        network_terms = [
            r'\b(router|switch|firewall|gateway|hub|bridge|access point|ap|server|database|load balancer)\b',
            r'\b(eth|gi|fa|se|lo|en|wlan)\d+',
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',  # IP addresses
            r'\b(vlan|subnet|network|interface|port|connection)\b'
        ]
        
        # Flowchart terms patterns
        flowchart_terms = [
            r'\b(start|end|begin|finish|stop|decision|process)\b',
            r'\b(yes|no|true|false|approve|reject)\b',
            r'\b(if|then|else|while|for|do|check|verify)\b',
            r'\?\s*$',  # Questions ending with ?
            r'\b(step|phase|stage)\s*\d+\b'
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
    
    def _get_search_components(self):
        """Initialize search components (lazy loading)."""
        if self._connection is None:
            from .database.data_access import DataAccess
            self._connection = Neo4jConnection(self.neo4j_uri, self.neo4j_user, self.neo4j_password)
            data_access = DataAccess(self._connection)
            schema_extractor = SchemaExtractor(self._connection)
            query_executor = QueryExecutor(self._connection)
            vector_search = VectorSearch(data_access, self.gemini_api_key)
            cypher_generator = CypherGenerator(self.gemini_api_key)
            self._retriever = TwoPhaseRetriever(
                vector_search, cypher_generator, 
                schema_extractor, query_executor, self._connection
            )
            self._configure_gemini()
    
    def search(self, query: str, diagram_id: str, method: str = None, top_k: int = None) -> Dict[str, Any]:
        """
        Search the knowledge graph with natural language.
        
        Args:
            query: Natural language search query
            diagram_id: Which diagram to search
            method: Search method - "vector", "cypher", or "auto" (defaults to DEFAULT_SEARCH_METHOD)
            top_k: Number of similar nodes to find (defaults to DEFAULT_TOP_K)
            
        Returns:
            Dict with search results containing 'nodes' and 'relationships' lists
        """
        method = method or self.DEFAULT_SEARCH_METHOD
        top_k = top_k or self.DEFAULT_TOP_K
            
        try:
            self._get_search_components()
            print(f"🔍 Searching: {query}")
            return self._retriever.retrieve(query, diagram_id, method, top_k)
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

    def invalidate_cache(self, diagram_id: str):
        """Invalidate search cache for a diagram (call after external data changes)"""
        self._get_search_components()
        self._retriever.vector_search.cache.invalidate_cache(diagram_id)
    
    def query_and_visualize(self, natural_query: str, diagram_id: str, 
                           backend: str = "graphviz",
                           layout: str = None,
                           output_path: Optional[str] = None,
                           include_explanation: bool = False,
                           detailed_explanation: bool = False,
                           method: str = "auto",
                           show_node_properties: bool = True,
                           show_edge_properties: bool = True,
                           generate_property_summary: bool = False) -> Dict[str, Any]:
        """
        Complete workflow: query + visualization + explanation
        
        Args:
            natural_query: Natural language query
            diagram_id: Specific diagram to search
            backend: "graphviz" or "networkx"
            layout: Layout algorithm - NetworkX: "spring", "circular", "shell"; Graphviz: "dot", "neato", "fdp", "circo"
            output_path: Custom output path for PNG image
            include_explanation: Generate natural language explanation
            detailed_explanation: Use detailed explanation format
            method: "auto", "vector", or "cypher"
            show_node_properties: Include all node properties in visualization
            show_edge_properties: Include all relationship properties in visualization
            generate_property_summary: Generate text summary of all properties
        
        Returns:
            Dict with nodes, relationships, image_path (or None for Jupyter), explanation, property_summary
        """
        
        # Step 1: Query graph data (search method handles its own logging)
        graph_data = self.search(natural_query, diagram_id, method)
        if "error" in graph_data:
            return graph_data
        
        if not graph_data.get("nodes"):
            return {"error": "No nodes found for visualization"}
        
        # Step 2: Generate visualization
        if not output_path:
            # Use hash of query to create short, unique filename
            query_hash = hashlib.md5(natural_query.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/visualizations/{diagram_id}_subgraph_{query_hash}_{timestamp}.png"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self._ensure_visualizer_initialized()
        
        # Generate visualization based on backend
        self._visualizer.switch_backend(backend)
        
        # Set appropriate default layout based on backend
        if not layout:
            layout = "dot" if backend == "graphviz" else "spring"
        
        if backend == "graphviz":
            image_path = self._visualizer.generate_image(
                graph_data["nodes"], 
                graph_data["relationships"], 
                output_path.replace('.png', ''),
                layout=layout,
                show_node_properties=show_node_properties,
                show_edge_properties=show_edge_properties
            )
        else:  # NetworkX
            image_path = self._visualizer.generate_image(
                graph_data["nodes"], 
                graph_data["relationships"], 
                output_path,
                layout=layout,
                show_node_properties=show_node_properties,
                show_edge_properties=show_edge_properties
            )
        
        
        # Check for visualization failure
        if not image_path:
            return {**graph_data, "error": "Failed to generate graph image"}
        
        # Step 3: Generate explanation
        explanation = ""
        if include_explanation:
            explanation = self.explain_subgraph(
                graph_data["nodes"],
                graph_data["relationships"], 
                natural_query,
                detailed_explanation
            )
        
        # Step 4: Generate property summary if requested
        property_summary = ""
        if generate_property_summary:
            property_summary = self._visualizer.create_property_summary(
                graph_data["nodes"],
                graph_data["relationships"]
            )
        
        result = {
            "nodes": graph_data["nodes"],
            "relationships": graph_data["relationships"],
            "image_path": image_path,
            "explanation": explanation,
            "method": graph_data.get("method", method)
        }
        
        if property_summary:
            result["property_summary"] = property_summary
        
        return result
    
    def explain_subgraph(self, nodes, relationships, original_query: str, detailed: bool = False) -> str:
        """Generate natural language explanation of the subgraph."""
        try:
            if not self.gemini_api_key:
                return "Explanation not available (Gemini API key required)"
            
            self._configure_gemini()
            model = genai.GenerativeModel(self.DEFAULT_GEMINI_MODEL)
            
            # Format nodes and relationships for explanation
            nodes_text = self._format_nodes_for_explanation(nodes)
            rels_text = self._format_relationships_for_explanation(relationships, nodes)
            
            # Generate appropriate prompt based on diagram type and detail level
            diagram_type = self._detect_diagram_type(nodes, relationships)
            prompt = self._create_explanation_prompt(
                diagram_type, original_query, nodes_text, rels_text, detailed
            )
            
            response = model.generate_content(prompt)
            return response.text if response else "Could not generate explanation"
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def _create_explanation_prompt(self, diagram_type: str, query: str, 
                                 nodes_text: list, rels_text: list, detailed: bool) -> str:
        """Create explanation prompt based on diagram type and detail level."""
        elements_text = chr(10).join(nodes_text)
        relationships_text = chr(10).join(rels_text)
        
        if detailed:
            return self._create_detailed_prompt(diagram_type, query, elements_text, relationships_text)
        else:
            return self._create_simple_prompt(diagram_type, query, elements_text, relationships_text)
    
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
            Explain this network diagram subgraph in simple terms:
            
            Query: "{query}"
            
            Network components:
            {elements_text}
            
            Network connections:
            {relationships_text}
            
            Provide a clear, concise explanation of the network topology and why it's relevant to the query.
            """,
            
            'flowchart': f"""
            Explain this process flowchart in simple terms:
            
            Query: "{query}"
            
            Process steps:
            {elements_text}
            
            Process flow:
            {relationships_text}
            
            Provide a clear, concise explanation of the workflow and why it's relevant to the query.
            """,
            
            'mixed': f"""
            Explain this diagram in simple terms:
            
            Query: "{query}"
            
            Elements:
            {elements_text}
            
            Relationships:
            {relationships_text}
            
            Provide a clear, concise explanation of what this shows and why it's relevant to the query.
            """
        }
        
        return prompts.get(diagram_type, prompts['mixed'])
    
    def close(self):
        """Clean up resources."""
        # GraphRAG client only manages its own connections
        if self._connection:
            self._connection.close()



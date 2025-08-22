"""
Root orchestration client for netbot-v2 system.

This client coordinates all modules:
- diagram_processing: Convert images to knowledge graphs
- embeddings: Add semantic search capabilities  
- graph_rag: Query and visualize knowledge graphs

Usage:
    from client import NetBot
    
    # Complete workflow
    netbot = NetBot()
    results = netbot.quickstart("diagram.png", "diagram_001", "find servers")
    
    # Step-by-step workflow
    netbot.process_diagram("diagram.png", "diagram_001")
    netbot.add_embeddings("diagram_001")
    results = netbot.search("find servers", "diagram_001")
"""

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Module imports
from diagram_processing.client import DiagramProcessor
from embeddings.client import EmbeddingManager
from graph_rag.client import GraphRAG

# Load environment variables
load_dotenv()


class NetBot:
    """
    Root orchestration client for the netbot-v2 system.
    
    Coordinates diagram processing, embeddings, and graph querying
    while maintaining clean separation between modules.
    """
    
    def __init__(self,
                 gemini_api_key: str = None,
                 neo4j_uri: str = None,
                 neo4j_user: str = None,
                 neo4j_password: str = None):
        """
        Initialize NetBot with credentials for all modules.
        
        Args:
            gemini_api_key: Gemini API key (defaults to env GEMINI_API_KEY)
            neo4j_uri: Neo4j database URI (defaults to env NEO4J_URI)
            neo4j_user: Neo4j username (defaults to env NEO4J_USER)
            neo4j_password: Neo4j password (defaults to env NEO4J_PASSWORD)
        """
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
        
        # Validate required credentials
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD must be provided via parameter or environment variable")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be provided via parameter or environment variable")
    
    def quickstart(self, image_path: str, query: str, 
                   explanation_detail: str = "basic") -> Dict[str, Any]:
        """
        Complete automated workflow: process + embeddings + search + visualization + explanation.
        
        Args:
            image_path: Path to diagram image
            query: Natural language search query
            explanation_detail: Explanation detail level - "none", "basic", or "detailed"
            
        Returns:
            Dict with comprehensive results:
            - 'nodes': List of relevant GraphNode objects
            - 'relationships': List of relevant GraphRelationship objects  
            - 'image_path': Path to generated visualization image
            - 'explanation': AI-generated explanation of the subgraph
            - 'error': Error message if workflow failed
        """
        try:
            print(f"🚀 Starting quickstart workflow for {image_path}")
            
            # Step 1: Process diagram (diagram_id will be auto-generated)
            print("📋 Processing diagram...")
            process_result = self.process_diagram(image_path)
            if process_result.get('status') != 'success':
                return {"error": f"Diagram processing failed: {process_result.get('error')}"}
            
            # Extract the diagram_id from the processing result
            diagram_id = process_result.get('diagram_id')
            if not diagram_id:
                # Fallback: generate from filename
                import os
                diagram_id = os.path.splitext(os.path.basename(image_path))[0]
            
            print(f"✅ Processed: {len(process_result.get('nodes', []))} nodes, {len(process_result.get('relationships', []))} relationships")
            print(f"📝 Using diagram_id: {diagram_id}")
            
            # Step 2: Add embeddings
            print("🧠 Adding embeddings...")
            embeddings_success = self.add_embeddings(diagram_id)
            if embeddings_success:
                print("✅ Embeddings added successfully")
            else:
                print("⚠️ Embeddings failed, continuing with basic search...")
            
            # Step 3: Search with visualization and explanation
            print(f"🔍 Searching and analyzing: {query}")
            
            # Configure explanation settings
            include_explanation = explanation_detail != "none"
            detailed_explanation = explanation_detail == "detailed"
            
            final_results = self.query_and_visualize(
                query=query,
                diagram_id=diagram_id,
                backend="graphviz",
                include_explanation=include_explanation,
                detailed_explanation=detailed_explanation
            )
            
            if not final_results.get('nodes'):
                return {"error": "No results found for the query"}
            
            print(f"✅ Found {len(final_results['nodes'])} nodes")
            if final_results.get('image_path'):
                print(f"📊 Visualization saved: {final_results['image_path']}")
            
            return final_results
            
        except Exception as e:
            return {"error": f"Quickstart workflow failed: {str(e)}"}
    
    def process_diagram(self, image_path: str, 
                       output_dir: str = "data/processed", 
                       force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process diagram image into knowledge graph.
        
        Args:
            image_path: Path to diagram image
            output_dir: Output directory for processed files
            force_reprocess: If True, reprocess even if diagram exists in Neo4j
            
        Returns:
            Dict with processing results or error information
        """
        # Auto-generate diagram_id from filename
        import os
        diagram_id = os.path.splitext(os.path.basename(image_path))[0]
        
        processor = DiagramProcessor(
            gemini_api_key=self.gemini_api_key,
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password
        )
        
        try:
            result = processor.process_diagram(
                image_path=image_path,
                diagram_id=diagram_id,
                output_dir=output_dir,
                force_reprocess=force_reprocess
            )
            return result
        finally:
            processor.close()
    
    def add_embeddings(self, diagram_id: str, batch_size: int = 100) -> bool:
        """
        Add semantic embeddings to diagram for better search.
        
        Args:
            diagram_id: Diagram to add embeddings to
            batch_size: Number of nodes to process at once
            
        Returns:
            True if successful, False otherwise
        """
        embedding_manager = EmbeddingManager(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password
        )
        
        try:
            return embedding_manager.add_embeddings(diagram_id, batch_size)
        finally:
            embedding_manager.close()
    
    def search(self, query: str, diagram_id: str, method: str = "auto", 
              top_k: int = 8) -> Dict[str, Any]:
        """
        Search knowledge graph with natural language.
        
        Args:
            query: Natural language search query
            diagram_id: Which diagram to search
            method: Search method - "vector", "cypher", or "auto"
            top_k: Number of similar nodes to find
            
        Returns:
            Dict with search results
        """
        rag = GraphRAG(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password,
            gemini_api_key=self.gemini_api_key
        )
        
        try:
            return rag.search(query, diagram_id, method, top_k)
        finally:
            rag.close()
    
    def query_and_visualize(self, query: str, diagram_id: str, 
                           backend: str = "graphviz", layout: str = None,
                           output_path: str = None, include_explanation: bool = False,
                           **kwargs) -> Dict[str, Any]:
        """
        Search and create visualization with explanation.
        
        Args:
            query: Natural language search query
            diagram_id: Which diagram to search
            backend: Visualization backend ("graphviz" or "networkx")
            layout: Layout algorithm (backend-specific)
            output_path: Custom output path for image
            include_explanation: Generate AI explanation
            **kwargs: Additional visualization options
            
        Returns:
            Dict with nodes, relationships, image_path, explanation
        """
        rag = GraphRAG(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password,
            gemini_api_key=self.gemini_api_key
        )
        
        try:
            return rag.query_and_visualize(
                natural_query=query,
                diagram_id=diagram_id,
                backend=backend,
                layout=layout,
                output_path=output_path,
                include_explanation=include_explanation,
                **kwargs
            )
        finally:
            rag.close()
    
    # Bulk Operations - Orchestration of module-specific bulk functionality
    
    def bulk_add_embeddings(self, diagram_ids: List[str], batch_size: int = 100) -> Dict[str, bool]:
        """
        Add embeddings to multiple diagrams in bulk.
        
        Args:
            diagram_ids: List of diagram IDs to process
            batch_size: Number of nodes to process at once per diagram
            
        Returns:
            Dict mapping diagram_id to success status
        """
        embedding_manager = EmbeddingManager(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password
        )
        
        try:
            return embedding_manager.bulk_add_embeddings(diagram_ids, batch_size)
        finally:
            embedding_manager.close()
    
    def bulk_check_embeddings(self, diagram_ids: List[str]) -> Dict[str, bool]:
        """
        Check embedding status for multiple diagrams.
        
        Args:
            diagram_ids: List of diagram IDs to check
            
        Returns:
            Dict mapping diagram_id to embedding existence status
        """
        embedding_manager = EmbeddingManager(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password
        )
        
        try:
            return embedding_manager.bulk_check_embeddings(diagram_ids)
        finally:
            embedding_manager.close()
    
    def get_diagrams_without_embeddings(self) -> List[str]:
        """
        Get list of diagrams that don't have embeddings yet.
        
        Returns:
            List of diagram IDs without embeddings
        """
        embedding_manager = EmbeddingManager(
            neo4j_uri=self.neo4j_uri,
            neo4j_user=self.neo4j_user,
            neo4j_password=self.neo4j_password
        )
        
        try:
            return embedding_manager.get_diagrams_without_embeddings()
        finally:
            embedding_manager.close()
    
    def bulk_quickstart(self, image_directory: str, query: str = None) -> Dict[str, Any]:
        """
        Bulk workflow: process all images in directory + add embeddings.
        Query/visualization should be done individually per diagram after bulk processing.
        
        Args:
            image_directory: Directory containing diagram images
            query: Optional query hint for guidance (not used for actual search)
            
        Returns:
            Dict with processing summary and list of processed diagrams
        """
        from pathlib import Path
        
        try:
            print(f"🚀 Starting bulk workflow for directory: {image_directory}")
            
            # Step 1: Bulk process all images in directory
            print("📋 Processing all diagrams...")
            processor = DiagramProcessor(
                gemini_api_key=self.gemini_api_key,
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password
            )
            
            try:
                # Use existing batch processing capability
                batch_results = processor.batch_process(
                    input_dir=image_directory,
                    output_dir="data/processed",
                    store_neo4j=True
                )
                
                if batch_results.get('status') == 'error':
                    return {"error": f"Batch processing failed: {batch_results.get('message')}"}
                
                processed_diagrams = batch_results.get('diagram_ids', [])
                print(f"✅ Processed {len(processed_diagrams)} diagrams")
                
            finally:
                processor.close()
            
            if not processed_diagrams:
                return {"error": "No diagrams were successfully processed"}
            
            # Step 2: Bulk add embeddings
            print("🧠 Adding embeddings to all diagrams...")
            embedding_results = self.bulk_add_embeddings(processed_diagrams)
            
            successful_embeddings = [diagram_id for diagram_id, success in embedding_results.items() if success]
            print(f"✅ Added embeddings to {len(successful_embeddings)} diagrams")
            
            print(f"✅ Bulk processing completed successfully!")
            print(f"📊 Summary:")
            print(f"  • Processed: {len(processed_diagrams)} diagrams")
            print(f"  • Embeddings: {len(successful_embeddings)} diagrams")
            print(f"\n💡 You can now search individual diagrams using:")
            for diagram_id in successful_embeddings:
                print(f"  netbot.search('{query}', '{diagram_id}')")
            
            return {
                "diagrams_processed": processed_diagrams,
                "diagrams_with_embeddings": successful_embeddings,
                "summary": {
                    "processed_count": len(processed_diagrams),
                    "embedding_count": len(successful_embeddings)
                },
                "message": "Bulk processing completed. Use individual search/visualization methods for querying."
            }
            
        except Exception as e:
            return {"error": f"Bulk quickstart workflow failed: {str(e)}"}



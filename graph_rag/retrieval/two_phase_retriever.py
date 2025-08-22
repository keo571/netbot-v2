"""
Two-phase retrieval strategy combining vector search with graph connectivity.
"""

from typing import Dict, List
from ..database.query_executor import ResultParser


class TwoPhaseRetriever:
    """Two-phase retrieval using vector similarity and graph connectivity"""
    
    def __init__(self, vector_search, cypher_generator, schema_extractor, query_executor, connection):
        self.vector_search = vector_search
        self.cypher_generator = cypher_generator
        self.schema_extractor = schema_extractor
        self.query_executor = query_executor
        self.connection = connection
    
    def retrieve(self, query: str, diagram_id: str, method: str = "auto", node_top_k: int = 8) -> Dict[str, List]:
        """Main retrieval method - chooses between vector and Cypher approaches"""
        
        if method == "vector":
            return self.retrieve_vector_only(query, diagram_id, node_top_k)
        elif method == "cypher":
            return self.retrieve_cypher_only(query, diagram_id)
        else:  # method == "auto"
            # Try vector search first (fast)
            if self.vector_search._check_embeddings_available(diagram_id):
                try:
                    result = self.vector_search.search_semantic_subgraph(query, diagram_id, node_top_k)
                    if result.get("nodes"):  # Got good results
                        return result
                except Exception as e:
                    print(f"Vector search failed: {e}, falling back to Cypher")
            
            # Fallback to traditional Cypher (reliable)
            print("⚠️ Two-phase search not available. Falling back to traditional search.")
            return self._execute_cypher_retrieval(query, diagram_id, "Cypher fallback")
    
    def retrieve_vector_only(self, query: str, diagram_id: str, node_top_k: int = 8) -> Dict[str, List]:
        """Force vector-only retrieval"""
        if not self.vector_search._check_embeddings_available(diagram_id):
            raise ValueError("Vector search not available. Embeddings not enabled.")
        
        return self.vector_search.search_semantic_subgraph(query, diagram_id, node_top_k)
    
    def retrieve_cypher_only(self, query: str, diagram_id: str) -> Dict[str, List]:
        """Force Cypher-only retrieval"""
        return self._execute_cypher_retrieval(query, diagram_id, "Cypher-only retrieval")
    
    def _execute_cypher_retrieval(self, query: str, diagram_id: str, operation_name: str) -> Dict[str, List]:
        """Common Cypher retrieval logic used by both fallback and Cypher-only modes"""
        try:
            with self.connection.get_session() as session:
                # Get schema info
                schema_info = self.schema_extractor.extract_schema(session, diagram_id)
                
                # Generate and execute Cypher query
                cypher_query = self.cypher_generator.generate_cypher_query(query, diagram_id, schema_info)
                result = self.query_executor.execute_with_timing(
                    session, cypher_query, operation_name="Cypher execution"
                )
                return ResultParser.parse_query_results(result)
        except Exception as e:
            print(f"❌ Error in {operation_name}: {e}")
            return {"nodes": [], "relationships": []}
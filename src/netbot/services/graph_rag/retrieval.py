"""
Two-phase retrieval system for GraphRAG.

Combines vector similarity search with graph traversal for comprehensive results.
"""

import time
from typing import List, Dict, Any, Tuple, Optional

from ...shared import (
    get_logger, EmbeddingClient, SearchResult as SharedSearchResult, 
    RetrievalMetadata, GraphNode, GraphRelationship
)
from .repository import GraphRAGRepository


class TwoPhaseRetriever:
    """
    Two-phase retrieval system combining vector and graph search.
    
    Phase 1: Vector similarity search to find relevant nodes
    Phase 2: Graph traversal to expand context and find relationships
    """
    
    def __init__(self, 
                 embedding_client: EmbeddingClient,
                 repository: GraphRAGRepository):
        """
        Initialize the retriever.
        
        Args:
            embedding_client: Shared embedding client
            repository: GraphRAG repository
        """
        self.logger = get_logger(__name__)
        self.embedding_client = embedding_client
        self.repository = repository
    
    def retrieve(self,
                query: str,
                diagram_id: str,
                method: str = "auto",
                top_k: int = 8,
                min_similarity: float = 0.1) -> SharedSearchResult:
        """
        Perform two-phase retrieval.
        
        Args:
            query: Natural language query
            diagram_id: Target diagram ID
            method: Retrieval method ("vector", "cypher", "auto", "hybrid")
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            Search result with nodes, relationships, and metadata
        """
        start_time = time.time()
        metadata = RetrievalMetadata()
        
        try:
            if method == "vector":
                nodes, relationships = self._vector_search(
                    query, diagram_id, top_k, min_similarity
                )
                metadata.vector_search_time_ms = (time.time() - start_time) * 1000
                
            elif method == "cypher":
                nodes, relationships = self._cypher_search(
                    query, diagram_id, top_k
                )
                metadata.graph_search_time_ms = (time.time() - start_time) * 1000
                
            elif method == "hybrid" or method == "auto":
                # Combine both approaches
                vector_start = time.time()
                vector_nodes, vector_rels = self._vector_search(
                    query, diagram_id, top_k // 2, min_similarity
                )
                metadata.vector_search_time_ms = (time.time() - vector_start) * 1000
                
                graph_start = time.time()
                graph_nodes, graph_rels = self._graph_expansion(
                    vector_nodes, diagram_id, max_expand=top_k // 2
                )
                metadata.graph_search_time_ms = (time.time() - graph_start) * 1000
                
                # Combine and deduplicate results
                nodes = self._merge_nodes(vector_nodes, graph_nodes)
                relationships = self._merge_relationships(vector_rels, graph_rels)
            
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
            
            # Calculate quality metrics
            metadata.total_candidates = len(nodes)
            metadata.filtered_results = len(nodes)  # After similarity filtering
            metadata.average_similarity = self._calculate_average_similarity(nodes)
            metadata.sources_used = ["neo4j", "embeddings"]
            metadata.is_complete = True
            
            # Create search result
            from ...shared.models.retrieval import Query
            query_obj = Query(
                text=query,
                diagram_id=diagram_id,
                top_k=top_k,
                min_similarity=min_similarity
            )
            
            result = SharedSearchResult(
                query=query_obj,
                nodes=nodes,
                relationships=relationships,
                metadata=metadata,
                relevance_score=metadata.average_similarity,
                confidence_score=self._calculate_confidence(nodes, relationships)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            metadata.is_complete = False
            metadata.has_warnings = True
            metadata.warnings.append(str(e))
            
            # Return empty result with error info
            from ...shared.models.retrieval import Query
            query_obj = Query(text=query, diagram_id=diagram_id)
            
            return SharedSearchResult(
                query=query_obj,
                nodes=[],
                relationships=[],
                metadata=metadata,
                relevance_score=0.0,
                confidence_score=0.0
            )
        
        finally:
            metadata.search_time_ms = (time.time() - start_time) * 1000
    
    def _vector_search(self,
                      query: str,
                      diagram_id: str,
                      top_k: int,
                      min_similarity: float) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """
        Phase 1: Vector similarity search.
        
        Args:
            query: Search query
            diagram_id: Target diagram
            top_k: Number of results
            min_similarity: Similarity threshold
            
        Returns:
            Tuple of (nodes, relationships)
        """
        try:
            # Get cached embedding matrix for the diagram
            embedding_matrix = self.embedding_client.get_embedding_matrix(diagram_id)
            
            if embedding_matrix is None:
                # Create embeddings if not cached
                self.logger.info(f"Creating embeddings for diagram: {diagram_id}")
                node_texts = self.repository.get_all_nodes_with_embeddings(diagram_id)
                
                if not node_texts:
                    return [], []
                
                texts = [text for _, text in node_texts]
                embedding_matrix = self.embedding_client.cache_diagram_embeddings(
                    diagram_id, texts
                )
            
            # Encode the query
            query_embedding = self.embedding_client.encode_single_text(query)
            
            # Find most similar nodes
            node_similarities = self.embedding_client.find_most_similar(
                query_embedding, embedding_matrix, top_k=top_k
            )
            
            # Filter by similarity threshold
            filtered_similarities = [
                (idx, sim) for idx, sim in node_similarities 
                if sim >= min_similarity
            ]
            
            # Get the actual nodes
            all_nodes = self.repository.list_by_diagram(diagram_id, limit=1000)
            selected_nodes = []
            
            for idx, similarity in filtered_similarities[:top_k]:
                if idx < len(all_nodes):
                    node = all_nodes[idx]
                    # Store similarity as metadata
                    node.metadata = node.metadata or {}
                    node.metadata['similarity_score'] = similarity
                    selected_nodes.append(node)
            
            # Get relationships between selected nodes
            if selected_nodes:
                node_ids = [node.id for node in selected_nodes]
                _, relationships = self.repository.get_subgraph(
                    node_ids, diagram_id, include_relationships=True
                )
            else:
                relationships = []
            
            return selected_nodes, relationships
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return [], []
    
    def _cypher_search(self,
                      query: str,
                      diagram_id: str,
                      top_k: int) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """
        Cypher-based search (placeholder implementation).
        
        In a full implementation, this would use an AI model to generate
        Cypher queries from natural language.
        """
        try:
            # This is a simplified implementation
            # In production, you'd use the model client to generate Cypher
            
            # For now, return a subset of nodes based on simple text matching
            all_nodes = self.repository.list_by_diagram(diagram_id, limit=100)
            
            # Simple keyword matching
            query_words = query.lower().split()
            matching_nodes = []
            
            for node in all_nodes:
                node_text = f"{node.label} {node.type}".lower()
                if any(word in node_text for word in query_words):
                    matching_nodes.append(node)
            
            # Limit results
            selected_nodes = matching_nodes[:top_k]
            
            # Get relationships
            if selected_nodes:
                node_ids = [node.id for node in selected_nodes]
                _, relationships = self.repository.get_subgraph(
                    node_ids, diagram_id, include_relationships=True
                )
            else:
                relationships = []
            
            return selected_nodes, relationships
            
        except Exception as e:
            self.logger.error(f"Cypher search failed: {e}")
            return [], []
    
    def _graph_expansion(self,
                       seed_nodes: List[GraphNode],
                       diagram_id: str,
                       max_expand: int) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """
        Phase 2: Expand context through graph traversal.
        
        Args:
            seed_nodes: Initial nodes from vector search
            diagram_id: Target diagram
            max_expand: Maximum nodes to add through expansion
            
        Returns:
            Tuple of (expanded_nodes, relationships)
        """
        if not seed_nodes:
            return [], []
        
        try:
            expanded_nodes = []
            all_relationships = []
            
            for seed_node in seed_nodes:
                # Get neighbors
                neighbors, rels = self.repository.get_node_neighbors(
                    seed_node.id, diagram_id, direction="both", max_hops=1
                )
                
                # Add neighbors (up to the limit)
                remaining_slots = max_expand - len(expanded_nodes)
                if remaining_slots > 0:
                    expanded_nodes.extend(neighbors[:remaining_slots])
                
                all_relationships.extend(rels)
            
            # Remove duplicates
            expanded_nodes = self._deduplicate_nodes(expanded_nodes)
            all_relationships = self._deduplicate_relationships(all_relationships)
            
            return expanded_nodes, all_relationships
            
        except Exception as e:
            self.logger.error(f"Graph expansion failed: {e}")
            return [], []
    
    def _merge_nodes(self, *node_lists: List[GraphNode]) -> List[GraphNode]:
        """Merge multiple node lists, removing duplicates."""
        seen_ids = set()
        merged = []
        
        for node_list in node_lists:
            for node in node_list:
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    merged.append(node)
        
        return merged
    
    def _merge_relationships(self, *rel_lists: List[GraphRelationship]) -> List[GraphRelationship]:
        """Merge multiple relationship lists, removing duplicates."""
        seen_ids = set()
        merged = []
        
        for rel_list in rel_lists:
            for rel in rel_list:
                if rel.id not in seen_ids:
                    seen_ids.add(rel.id)
                    merged.append(rel)
        
        return merged
    
    def _deduplicate_nodes(self, nodes: List[GraphNode]) -> List[GraphNode]:
        """Remove duplicate nodes based on ID."""
        seen_ids = set()
        deduplicated = []
        
        for node in nodes:
            if node.id not in seen_ids:
                seen_ids.add(node.id)
                deduplicated.append(node)
        
        return deduplicated
    
    def _deduplicate_relationships(self, relationships: List[GraphRelationship]) -> List[GraphRelationship]:
        """Remove duplicate relationships based on ID."""
        seen_ids = set()
        deduplicated = []
        
        for rel in relationships:
            if rel.id not in seen_ids:
                seen_ids.add(rel.id)
                deduplicated.append(rel)
        
        return deduplicated
    
    def _calculate_average_similarity(self, nodes: List[GraphNode]) -> float:
        """Calculate average similarity score from nodes."""
        if not nodes:
            return 0.0
        
        similarities = []
        for node in nodes:
            if node.metadata and 'similarity_score' in node.metadata:
                similarities.append(node.metadata['similarity_score'])
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_confidence(self, 
                            nodes: List[GraphNode], 
                            relationships: List[GraphRelationship]) -> float:
        """Calculate overall confidence score."""
        all_confidences = []
        all_confidences.extend([node.confidence_score for node in nodes])
        all_confidences.extend([rel.confidence_score for rel in relationships])
        
        return sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
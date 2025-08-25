"""
Vector similarity search for graph elements.
Pure query engine - assumes embeddings already exist in Neo4j.
"""

import numpy as np
from typing import List, Dict

from models.graph_models import GraphNode
from embeddings.embedding_encoder import EmbeddingEncoder
from .embedding_cache import EmbeddingCache


class EmbeddingsNotFoundError(Exception):
    """Raised when no embeddings are found for a diagram"""
    pass


class VectorSearchCache:
    """Global cache for VectorSearch components"""
    _embedding_encoder = None
    _embedding_caches = {}  # diagram_id -> EmbeddingCache
    
    @classmethod
    def get_embedding_encoder(cls):
        """Get or create cached embedding encoder"""
        if cls._embedding_encoder is None:
            try:
                cls._embedding_encoder = EmbeddingEncoder()
                print(f"✅ VectorSearch: Cached EmbeddingEncoder initialized")
            except Exception as e:
                print(f"❌ VectorSearch: Failed to initialize EmbeddingEncoder: {e}")
                import traceback
                traceback.print_exc()
                cls._embedding_encoder = None
        return cls._embedding_encoder
    
    @classmethod
    def get_embedding_cache(cls, data_access, cache_key):
        """Get or create cached embedding cache"""
        if cache_key not in cls._embedding_caches:
            cls._embedding_caches[cache_key] = EmbeddingCache(data_access)
        return cls._embedding_caches[cache_key]


class VectorSearch:
    """Handles vector similarity search operations - pure query engine"""
    
    def __init__(self, data_access, gemini_api_key=None):
        self.data_access = data_access
        self.gemini_api_key = gemini_api_key
        # Use cached encoder instead of creating new one
        self.embedding_encoder = VectorSearchCache.get_embedding_encoder()
        # Use cached embedding cache
        cache_key = f"{id(data_access)}"  # Use data_access instance as cache key
        self.cache = VectorSearchCache.get_embedding_cache(data_access, cache_key)
    
    def cosine_similarity_vectorized(self, query_embedding: np.ndarray, embedding_matrix: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and multiple embeddings efficiently"""
        if embedding_matrix.size == 0:
            return np.array([])
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(embedding_matrix.shape[0])
        query_normalized = query_embedding / query_norm
        
        # Normalize all embeddings in matrix
        embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
        # Avoid division by zero
        embedding_norms = np.where(embedding_norms == 0, 1, embedding_norms)
        embeddings_normalized = embedding_matrix / embedding_norms[:, np.newaxis]
        
        # Vectorized cosine similarity: single matrix multiplication
        similarities = np.dot(embeddings_normalized, query_normalized)
        
        # Set similarity to 0 for zero-norm embeddings
        zero_norm_mask = np.linalg.norm(embedding_matrix, axis=1) == 0
        similarities[zero_norm_mask] = 0.0
        
        return similarities
    
    def search_nodes(self, query: str, diagram_id: str, top_k: int = 8, min_similarity: float = 0.1) -> List[GraphNode]:
        """Phase 1: Search for nodes using vector similarity with caching and vectorization"""
        if not self.embedding_encoder:
            print("❌ Vector search not available. Embeddings not enabled.")
            return []
        
        # Get query embedding
        query_embedding = self.embedding_encoder.encode_query(query)
        
        # Get cached embeddings and node data (includes availability check)
        embedding_matrix, node_data = self.cache.get_cached_nodes(diagram_id)
        
        # Check if we have any nodes with embeddings
        if len(node_data) == 0:
            raise EmbeddingsNotFoundError(
                f"No embeddings found for diagram '{diagram_id}'. "
                f"To enable semantic search, add embeddings with: "
                f"python embeddings/cli.py add {diagram_id}"
            )
        
        # Vectorized similarity computation (much faster than loops)
        similarities = self.cosine_similarity_vectorized(query_embedding, embedding_matrix)
        
        # Get top-k indices using numpy's efficient sorting
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Reverse for descending order
        
        # Filter by minimum similarity threshold to exclude noise
        good_indices = top_indices[similarities[top_indices] >= min_similarity]
        
        # Graceful degradation: if all similarities are below threshold, keep the best match
        if len(good_indices) == 0 and len(top_indices) > 0:
            good_indices = top_indices[:1]  # Keep the single best match
            print(f"⚠️ All similarities below {min_similarity}, keeping best match (similarity: {similarities[top_indices[0]]:.3f})")
        
        # Build GraphNode objects from filtered node data
        top_nodes = []
        for idx in good_indices:
            data = node_data[idx]
            node = GraphNode(
                id=data['id'],
                label=data['label'],
                type=data['type'],
                diagram_id=diagram_id,
                properties=data['properties']
            )
            top_nodes.append(node)
        
        filtered_count = len(top_indices) - len(good_indices)
        if filtered_count > 0:
            print(f"Vector search: {len(good_indices)} quality nodes (filtered out {filtered_count} low-similarity)")
        else:
            print(f"Vector search found {len(top_nodes)} similar nodes")
        
        return top_nodes
    
    def search_semantic_subgraph(self, query: str, diagram_id: str, node_top_k: int = 8, min_similarity: float = 0.1) -> Dict[str, List]:
        """Find semantically relevant subgraph using vector similarity and graph connectivity"""
        print(f"Starting semantic subgraph search for: {query}")
        
        # Phase 1: Vector similarity search - find top-k most semantically similar nodes
        similar_nodes = self.search_nodes(query, diagram_id, node_top_k, min_similarity)
        print(f"Phase 1: Found {len(similar_nodes)} quality seed nodes")
        
        # Phase 2: Graph connectivity - build coherent subgraph from similar nodes
        subgraph = self._build_coherent_subgraph(similar_nodes, diagram_id)
        
        return subgraph
    
    def _build_coherent_subgraph(self, nodes: List[GraphNode], diagram_id: str) -> Dict[str, List]:
        """Phase 2: Build a coherent subgraph from similar nodes by finding paths and intermediate nodes"""
        
        # Start with the similar nodes
        result_nodes = list(nodes)
        node_ids = [node.id for node in nodes]
        
        # First try direct connections (fast)
        direct_rels = self.data_access.get_connecting_relationships(node_ids, diagram_id)
        
        # Check for orphaned nodes (nodes with no direct connections to others)
        connected_node_ids = set()
        for rel in direct_rels:
            connected_node_ids.add(rel.source_id)
            connected_node_ids.add(rel.target_id)
        
        orphaned_node_ids = [node_id for node_id in node_ids if node_id not in connected_node_ids]
        
        # If we have orphaned nodes, look for multi-hop paths to connect them
        all_rels = direct_rels
        if orphaned_node_ids and len(nodes) > 1:
            print(f"Found {len(orphaned_node_ids)} orphaned nodes, searching for multi-hop paths...")
            
            # Find paths between all nodes (including orphaned ones)
            path_rels = self.data_access.get_paths_between_nodes(node_ids, diagram_id, max_hops=3)
            
            # Find intermediate nodes in those paths
            intermediate_nodes = self.data_access.get_intermediate_nodes_in_paths(node_ids, diagram_id, max_hops=3)
            
            # Add intermediate nodes to result
            result_nodes.extend(intermediate_nodes)
            
            # Combine direct and path relationships
            all_rels = direct_rels + path_rels
            
            print(f"Built hybrid subgraph: {len(result_nodes)} nodes ({len(intermediate_nodes)} intermediate), {len(all_rels)} relationships")
        else:
            print(f"Built direct subgraph: {len(result_nodes)} nodes, {len(all_rels)} relationships")
        
        return {
            "nodes": result_nodes,
            "relationships": all_rels
        }
    
    def _check_embeddings_available(self, diagram_id: str) -> bool:
        """Check if embeddings exist for a diagram (all-or-nothing per diagram)"""
        try:
            with self.cache.data_access.connection.get_session() as session:
                result = session.run(
                    "MATCH (n {diagram_id: $diagram_id}) RETURN n.embedding IS NOT NULL as has_embedding LIMIT 1",
                    diagram_id=diagram_id
                )
                record = result.single()
                has_embeddings = record["has_embedding"] if record else False
                print(f"🔍 Embeddings check for {diagram_id}: {has_embeddings}")
                return has_embeddings
        except Exception as e:
            print(f"❌ Error checking embeddings: {e}")
            return False
    

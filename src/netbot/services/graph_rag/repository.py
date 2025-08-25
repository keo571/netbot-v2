"""
Repository layer for GraphRAG operations.

Handles specialized graph queries for search and retrieval operations.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ...shared import BaseRepository, GraphNode, GraphRelationship, DatabaseError


class GraphRAGRepository(BaseRepository):
    """
    Repository for GraphRAG database operations.
    
    Provides specialized queries for graph-based search and retrieval.
    """
    
    def create(self, entity) -> Any:
        """Required by base repository - not used directly in GraphRAG."""
        raise NotImplementedError("GraphRAG repository doesn't create entities directly")
    
    def get_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its ID."""
        query = """
        MATCH (n {id: $node_id})
        RETURN n
        """
        
        try:
            results = self.execute_query(query, {'node_id': node_id})
            if not results:
                return None
            
            return self._node_from_db(results[0]['n'])
            
        except Exception as e:
            raise DatabaseError(f"Failed to get node {node_id}: {e}")
    
    def list_by_diagram(self, diagram_id: str, limit: int = 100) -> List[GraphNode]:
        """List all nodes for a diagram."""
        query = """
        MATCH (n {diagram_id: $diagram_id})
        RETURN n
        ORDER BY n.created_at DESC
        LIMIT $limit
        """
        
        try:
            results = self.execute_query(query, {
                'diagram_id': diagram_id,
                'limit': limit
            })
            
            return [self._node_from_db(result['n']) for result in results]
            
        except Exception as e:
            raise DatabaseError(f"Failed to list nodes for diagram {diagram_id}: {e}")
    
    def update(self, entity) -> Any:
        """Required by base repository - not typically used in GraphRAG."""
        raise NotImplementedError("GraphRAG repository doesn't update entities directly")
    
    def get_all_nodes_with_embeddings(self, diagram_id: str) -> List[Tuple[GraphNode, str]]:
        """
        Get all nodes with their text content for embedding generation.
        
        Args:
            diagram_id: Target diagram ID
            
        Returns:
            List of (node, text_content) tuples
        """
        query = """
        MATCH (n {diagram_id: $diagram_id})
        RETURN n
        ORDER BY n.label
        """
        
        try:
            results = self.execute_query(query, {'diagram_id': diagram_id})
            
            node_texts = []
            for result in results:
                node = self._node_from_db(result['n'])
                
                # Create text representation for embedding
                text_parts = [node.label, node.type]
                
                # Add relevant properties
                for key, value in node.properties.items():
                    if key not in ['id', 'diagram_id', 'embedding'] and value:
                        text_parts.append(f"{key}: {value}")
                
                text_content = " ".join(str(part) for part in text_parts if part)
                node_texts.append((node, text_content))
            
            return node_texts
            
        except Exception as e:
            raise DatabaseError(f"Failed to get nodes with embeddings for {diagram_id}: {e}")
    
    def search_nodes_by_similarity(self,
                                  query_embedding: np.ndarray,
                                  diagram_id: str,
                                  top_k: int = 8) -> List[Tuple[GraphNode, float]]:
        """
        Search for nodes by embedding similarity.
        
        This is a placeholder - in a production system, you'd use a vector database
        or Neo4j's vector search capabilities.
        
        Args:
            query_embedding: Query embedding vector
            diagram_id: Target diagram ID
            top_k: Number of results to return
            
        Returns:
            List of (node, similarity_score) tuples
        """
        try:
            # Get all nodes for the diagram
            nodes = self.list_by_diagram(diagram_id, limit=1000)
            
            # This is a simplified version - in production, you'd:
            # 1. Store embeddings in the database or a vector store
            # 2. Use efficient similarity search
            # 3. Index the embeddings for fast retrieval
            
            # For now, return nodes with dummy similarity scores
            # This would be replaced with actual embedding-based search
            node_similarities = []
            for i, node in enumerate(nodes[:top_k]):
                # Placeholder similarity calculation
                similarity = max(0.1, 1.0 - (i * 0.1))  # Decreasing similarity
                node_similarities.append((node, similarity))
            
            return node_similarities
            
        except Exception as e:
            raise DatabaseError(f"Failed to search nodes by similarity: {e}")
    
    def get_subgraph(self,
                    node_ids: List[str],
                    diagram_id: str,
                    include_relationships: bool = True) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """
        Get a subgraph containing specified nodes and their relationships.
        
        Args:
            node_ids: List of node IDs to include
            diagram_id: Target diagram ID
            include_relationships: Whether to include relationships
            
        Returns:
            Tuple of (nodes, relationships)
        """
        if not node_ids:
            return [], []
        
        try:
            # Get nodes
            nodes_query = """
            MATCH (n {diagram_id: $diagram_id})
            WHERE n.id IN $node_ids
            RETURN n
            """
            
            nodes_results = self.execute_query(nodes_query, {
                'diagram_id': diagram_id,
                'node_ids': node_ids
            })
            
            nodes = [self._node_from_db(result['n']) for result in nodes_results]
            
            relationships = []
            if include_relationships:
                # Get relationships between the nodes
                rels_query = """
                MATCH (source {diagram_id: $diagram_id})-[r {diagram_id: $diagram_id}]->(target {diagram_id: $diagram_id})
                WHERE source.id IN $node_ids AND target.id IN $node_ids
                RETURN r, source.id as source_id, target.id as target_id
                """
                
                rels_results = self.execute_query(rels_query, {
                    'diagram_id': diagram_id,
                    'node_ids': node_ids
                })
                
                relationships = [
                    self._relationship_from_db(
                        result['r'], 
                        result['source_id'], 
                        result['target_id']
                    ) 
                    for result in rels_results
                ]
            
            return nodes, relationships
            
        except Exception as e:
            raise DatabaseError(f"Failed to get subgraph: {e}")
    
    def search_by_cypher(self, cypher_query: str, parameters: Dict[str, Any] = None) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """
        Execute a Cypher query and return nodes and relationships.
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Tuple of (nodes, relationships)
        """
        try:
            results = self.execute_query(cypher_query, parameters or {})
            
            nodes = []
            relationships = []
            
            for result in results:
                # Extract nodes from the result
                for key, value in result.items():
                    if isinstance(value, dict):
                        # Check if it's a node or relationship based on structure
                        if 'id' in value and 'label' in value and 'type' in value:
                            # Looks like a node
                            node = self._node_from_db(value)
                            if node not in nodes:
                                nodes.append(node)
                        elif 'source_id' in value and 'target_id' in value:
                            # Looks like a relationship
                            rel = self._relationship_from_db(
                                value, 
                                value['source_id'], 
                                value['target_id']
                            )
                            if rel not in relationships:
                                relationships.append(rel)
            
            return nodes, relationships
            
        except Exception as e:
            raise DatabaseError(f"Failed to execute Cypher query: {e}")
    
    def get_node_neighbors(self,
                          node_id: str,
                          diagram_id: str,
                          direction: str = "both",
                          max_hops: int = 1) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """
        Get neighboring nodes within specified hops.
        
        Args:
            node_id: Starting node ID
            diagram_id: Target diagram ID
            direction: "in", "out", or "both"
            max_hops: Maximum number of hops
            
        Returns:
            Tuple of (nodes, relationships)
        """
        direction_map = {
            "out": "->",
            "in": "<-",
            "both": "-"
        }
        
        direction_pattern = direction_map.get(direction, "-")
        
        query = f"""
        MATCH path = (start {{id: $node_id, diagram_id: $diagram_id}})
                     {direction_pattern}*1..{max_hops} 
                     (neighbor {{diagram_id: $diagram_id}})
        WITH nodes(path) as path_nodes, relationships(path) as path_rels
        UNWIND path_nodes as n
        UNWIND path_rels as r
        RETURN DISTINCT n, r
        """
        
        try:
            results = self.execute_query(query, {
                'node_id': node_id,
                'diagram_id': diagram_id
            })
            
            nodes = []
            relationships = []
            
            for result in results:
                if result.get('n'):
                    node = self._node_from_db(result['n'])
                    if node not in nodes:
                        nodes.append(node)
                
                if result.get('r'):
                    # This is simplified - in practice, you'd need source/target IDs
                    # from the path structure
                    pass
            
            return nodes, relationships
            
        except Exception as e:
            raise DatabaseError(f"Failed to get node neighbors: {e}")
    
    def search_nodes_by_properties(self,
                                 diagram_id: str,
                                 property_filters: Dict[str, Any],
                                 node_type: Optional[str] = None) -> List[GraphNode]:
        """
        Search nodes by property values.
        
        Args:
            diagram_id: Target diagram ID
            property_filters: Property key-value pairs to match
            node_type: Optional node type filter
            
        Returns:
            List of matching nodes
        """
        query_parts = ["MATCH (n {diagram_id: $diagram_id"]
        params = {'diagram_id': diagram_id}
        
        if node_type:
            query_parts.append(f", type: $node_type")
            params['node_type'] = node_type
        
        query_parts.append("})")
        
        # Add property filters
        if property_filters:
            conditions = []
            for key, value in property_filters.items():
                param_name = f"prop_{key}"
                conditions.append(f"n.{key} = ${param_name}")
                params[param_name] = value
            
            if conditions:
                query_parts.append("WHERE " + " AND ".join(conditions))
        
        query_parts.append("RETURN n")
        query = " ".join(query_parts)
        
        try:
            results = self.execute_query(query, params)
            return [self._node_from_db(result['n']) for result in results]
            
        except Exception as e:
            raise DatabaseError(f"Failed to search nodes by properties: {e}")
    
    def _node_from_db(self, node_data: Dict[str, Any]) -> GraphNode:
        """Convert database node data to GraphNode object."""
        # Extract metadata fields
        metadata = {}
        properties = {}
        
        for key, value in node_data.items():
            if key.startswith('meta_'):
                metadata[key[5:]] = value
            elif key not in ['id', 'label', 'type', 'diagram_id', 'confidence_score', 'source_type', 'created_at']:
                properties[key] = value
        
        return GraphNode(
            id=node_data['id'],
            label=node_data['label'],
            type=node_data['type'],
            diagram_id=node_data['diagram_id'],
            properties=properties,
            confidence_score=node_data.get('confidence_score', 1.0),
            source_type=node_data.get('source_type', 'diagram'),
            metadata=metadata
        )
    
    def _relationship_from_db(self,
                            rel_data: Dict[str, Any],
                            source_id: str,
                            target_id: str) -> GraphRelationship:
        """Convert database relationship data to GraphRelationship object."""
        # Extract metadata fields
        metadata = {}
        properties = {}
        
        for key, value in rel_data.items():
            if key.startswith('meta_'):
                metadata[key[5:]] = value
            elif key not in ['id', 'type', 'diagram_id', 'confidence_score', 'source_type', 'created_at']:
                properties[key] = value
        
        return GraphRelationship(
            id=rel_data['id'],
            source_id=source_id,
            target_id=target_id,
            type=rel_data['type'],
            diagram_id=rel_data['diagram_id'],
            properties=properties,
            confidence_score=rel_data.get('confidence_score', 1.0),
            source_type=rel_data.get('source_type', 'diagram'),
            metadata=metadata
        )
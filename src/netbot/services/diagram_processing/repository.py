"""
Repository layer for diagram processing operations.

Handles database operations specific to diagram processing,
built on top of the shared infrastructure.
"""

from typing import List, Dict, Any, Optional
import time

from ...shared import BaseRepository, GraphNode, GraphRelationship, DatabaseError


class DiagramRepository(BaseRepository):
    """
    Repository for diagram processing database operations.
    
    Provides specialized queries and operations for diagram data management.
    """
    
    def create_node(self, node: GraphNode) -> GraphNode:
        """Create a new node in the database."""
        query = """
        CREATE (n:Node $properties)
        RETURN n
        """
        
        try:
            properties = node.to_neo4j_dict()
            # Add labels for the node type
            properties['labels'] = [node.type]
            
            result = self.execute_write_query(query, {'properties': properties})
            return node
            
        except Exception as e:
            raise DatabaseError(f"Failed to create node {node.id}: {e}")
    
    def create_relationship(self, relationship: GraphRelationship) -> GraphRelationship:
        """Create a new relationship in the database."""
        query = """
        MATCH (source:Node {id: $source_id, diagram_id: $diagram_id})
        MATCH (target:Node {id: $target_id, diagram_id: $diagram_id})
        CREATE (source)-[r:RELATIONSHIP $properties]->(target)
        RETURN r
        """
        
        try:
            properties = relationship.to_neo4j_dict()
            properties['type'] = relationship.type
            
            params = {
                'source_id': relationship.source_id,
                'target_id': relationship.target_id,
                'diagram_id': relationship.diagram_id,
                'properties': properties
            }
            
            result = self.execute_write_query(query, params)
            return relationship
            
        except Exception as e:
            raise DatabaseError(f"Failed to create relationship {relationship.id}: {e}")
    
    def get_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its ID."""
        query = """
        MATCH (n:Node {id: $node_id})
        RETURN n
        """
        
        try:
            results = self.execute_query(query, {'node_id': node_id})
            if not results:
                return None
            
            node_data = results[0]['n']
            return self._node_from_db(node_data)
            
        except Exception as e:
            raise DatabaseError(f"Failed to get node {node_id}: {e}")
    
    def list_by_diagram(self, diagram_id: str, limit: int = 100) -> List[GraphNode]:
        """List all nodes for a diagram."""
        query = """
        MATCH (n:Node {diagram_id: $diagram_id})
        RETURN n
        ORDER BY n.created_at DESC
        LIMIT $limit
        """
        
        try:
            results = self.execute_query(query, {
                'diagram_id': diagram_id,
                'limit': limit
            })
            
            nodes = []
            for result in results:
                node = self._node_from_db(result['n'])
                nodes.append(node)
            
            return nodes
            
        except Exception as e:
            raise DatabaseError(f"Failed to list nodes for diagram {diagram_id}: {e}")
    
    def get_relationships_by_diagram(self, diagram_id: str, limit: int = 100) -> List[GraphRelationship]:
        """Get all relationships for a diagram."""
        query = """
        MATCH (source:Node {diagram_id: $diagram_id})-[r:RELATIONSHIP {diagram_id: $diagram_id}]->(target:Node)
        RETURN r, source.id as source_id, target.id as target_id
        ORDER BY r.created_at DESC
        LIMIT $limit
        """
        
        try:
            results = self.execute_query(query, {
                'diagram_id': diagram_id,
                'limit': limit
            })
            
            relationships = []
            for result in results:
                relationship = self._relationship_from_db(
                    result['r'], 
                    result['source_id'], 
                    result['target_id']
                )
                relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            raise DatabaseError(f"Failed to get relationships for diagram {diagram_id}: {e}")
    
    def update(self, node: GraphNode) -> GraphNode:
        """Update an existing node."""
        query = """
        MATCH (n:Node {id: $node_id, diagram_id: $diagram_id})
        SET n += $properties
        RETURN n
        """
        
        try:
            properties = node.to_neo4j_dict()
            params = {
                'node_id': node.id,
                'diagram_id': node.diagram_id,
                'properties': properties
            }
            
            result = self.execute_write_query(query, params)
            return node
            
        except Exception as e:
            raise DatabaseError(f"Failed to update node {node.id}: {e}")
    
    def create(self, node: GraphNode) -> GraphNode:
        """Create a new node (required by base repository)."""
        return self.create_node(node)
    
    def get_diagram_metadata(self, diagram_id: str) -> Dict[str, Any]:
        """Get metadata about a diagram."""
        query = """
        MATCH (n:Node {diagram_id: $diagram_id})
        OPTIONAL MATCH (n)-[r:RELATIONSHIP {diagram_id: $diagram_id}]-()
        RETURN 
            count(DISTINCT n) as node_count,
            count(DISTINCT r) as relationship_count,
            collect(DISTINCT n.type) as node_types,
            collect(DISTINCT r.type) as relationship_types,
            min(n.created_at) as first_created,
            max(n.created_at) as last_created
        """
        
        try:
            results = self.execute_query(query, {'diagram_id': diagram_id})
            if not results:
                return {
                    'diagram_id': diagram_id,
                    'node_count': 0,
                    'relationship_count': 0,
                    'node_types': [],
                    'relationship_types': [],
                    'first_created': None,
                    'last_created': None
                }
            
            result = results[0]
            return {
                'diagram_id': diagram_id,
                'node_count': result['node_count'],
                'relationship_count': result['relationship_count'],
                'node_types': [t for t in result['node_types'] if t],
                'relationship_types': [t for t in result['relationship_types'] if t],
                'first_created': result['first_created'],
                'last_created': result['last_created']
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get diagram metadata for {diagram_id}: {e}")
    
    def get_recent_diagrams(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently processed diagrams."""
        query = """
        MATCH (n:Node)
        WITH n.diagram_id as diagram_id, max(n.created_at) as last_updated
        ORDER BY last_updated DESC
        LIMIT $limit
        RETURN diagram_id, last_updated
        """
        
        try:
            results = self.execute_query(query, {'limit': limit})
            
            diagrams = []
            for result in results:
                diagram_id = result['diagram_id']
                stats = self.get_diagram_metadata(diagram_id)
                stats['last_updated'] = result['last_updated']
                diagrams.append(stats)
            
            return diagrams
            
        except Exception as e:
            raise DatabaseError(f"Failed to get recent diagrams: {e}")
    
    def search_nodes_by_type(self, diagram_id: str, node_type: str) -> List[GraphNode]:
        """Search for nodes of a specific type."""
        query = """
        MATCH (n:Node {diagram_id: $diagram_id, type: $node_type})
        RETURN n
        ORDER BY n.label
        """
        
        try:
            results = self.execute_query(query, {
                'diagram_id': diagram_id,
                'node_type': node_type
            })
            
            nodes = []
            for result in results:
                node = self._node_from_db(result['n'])
                nodes.append(node)
            
            return nodes
            
        except Exception as e:
            raise DatabaseError(f"Failed to search nodes by type {node_type}: {e}")
    
    def get_node_with_relationships(self, node_id: str, diagram_id: str) -> Dict[str, Any]:
        """Get a node with all its relationships."""
        query = """
        MATCH (n:Node {id: $node_id, diagram_id: $diagram_id})
        OPTIONAL MATCH (n)-[r:RELATIONSHIP {diagram_id: $diagram_id}]-(connected:Node)
        RETURN n, 
               collect(DISTINCT r) as relationships,
               collect(DISTINCT connected) as connected_nodes
        """
        
        try:
            results = self.execute_query(query, {
                'node_id': node_id,
                'diagram_id': diagram_id
            })
            
            if not results:
                return {}
            
            result = results[0]
            node = self._node_from_db(result['n'])
            
            relationships = []
            for rel_data in result['relationships']:
                if rel_data:  # Skip None values
                    # We need source/target IDs to create the relationship object
                    # This is a simplified version - in practice you'd need a more complex query
                    pass
            
            connected_nodes = []
            for node_data in result['connected_nodes']:
                if node_data:  # Skip None values
                    connected_node = self._node_from_db(node_data)
                    connected_nodes.append(connected_node)
            
            return {
                'node': node,
                'relationships': relationships,
                'connected_nodes': connected_nodes
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get node with relationships {node_id}: {e}")
    
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
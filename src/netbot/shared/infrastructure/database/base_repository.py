"""
Base repository pattern for database operations.

Provides a consistent interface for all database repositories in NetBot V2.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Type, TypeVar
from contextlib import contextmanager

from .connection_manager import get_database
from ...exceptions import DatabaseError
from ...models.base import BaseModel

T = TypeVar('T', bound=BaseModel)


class BaseRepository(ABC):
    """
    Abstract base repository for consistent database operations.
    
    Provides common patterns for CRUD operations and transaction management.
    """
    
    def __init__(self, connection_name: str = "default"):
        """
        Initialize repository with database connection.
        
        Args:
            connection_name: Database connection to use
        """
        self.db = get_database()
        self.connection_name = connection_name
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        with self.db.get_session(self.connection_name) as session:
            tx = session.begin_transaction()
            try:
                yield tx
                tx.commit()
            except Exception as e:
                tx.rollback()
                raise DatabaseError(f"Transaction failed: {e}")
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        try:
            results = self.db.execute_query(query, parameters, self.connection_name)
            return [record.data() for record in results]
        except Exception as e:
            raise DatabaseError(f"Query execution failed: {e}")
    
    def execute_write_query(self, query: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a write query with transaction handling.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query execution summary
        """
        def write_tx(tx, query_str, params):
            result = tx.run(query_str, params or {})
            summary = result.consume()
            return {
                'nodes_created': summary.counters.nodes_created,
                'relationships_created': summary.counters.relationships_created,
                'nodes_deleted': summary.counters.nodes_deleted,
                'relationships_deleted': summary.counters.relationships_deleted,
                'properties_set': summary.counters.properties_set,
            }
        
        return self.db.execute_write_transaction(
            write_tx, 
            query, 
            parameters,
            connection_name=self.connection_name
        )
    
    def count_nodes(self, label: str = None, filters: Dict[str, Any] = None) -> int:
        """
        Count nodes with optional label and property filters.
        
        Args:
            label: Node label to filter by
            filters: Property filters
            
        Returns:
            Node count
        """
        query_parts = ["MATCH (n"]
        params = {}
        
        if label:
            query_parts.append(f":{label}")
        
        query_parts.append(")")
        
        if filters:
            conditions = []
            for key, value in filters.items():
                param_name = f"filter_{key}"
                conditions.append(f"n.{key} = ${param_name}")
                params[param_name] = value
            
            if conditions:
                query_parts.append("WHERE " + " AND ".join(conditions))
        
        query_parts.append("RETURN count(n) as count")
        query = " ".join(query_parts)
        
        results = self.execute_query(query, params)
        return results[0]['count'] if results else 0
    
    def exists(self, node_id: str, label: str = None) -> bool:
        """
        Check if a node exists by ID.
        
        Args:
            node_id: Node ID to check
            label: Optional node label
            
        Returns:
            True if node exists
        """
        query_parts = ["MATCH (n"]
        if label:
            query_parts.append(f":{label}")
        query_parts.extend(["{id: $node_id})", "RETURN count(n) > 0 as exists"])
        
        query = " ".join(query_parts)
        results = self.execute_query(query, {'node_id': node_id})
        return results[0]['exists'] if results else False
    
    def delete_by_id(self, node_id: str, label: str = None) -> bool:
        """
        Delete a node by ID.
        
        Args:
            node_id: Node ID to delete
            label: Optional node label
            
        Returns:
            True if node was deleted
        """
        query_parts = ["MATCH (n"]
        if label:
            query_parts.append(f":{label}")
        query_parts.extend(["{id: $node_id})", "DETACH DELETE n"])
        
        query = " ".join(query_parts)
        summary = self.execute_write_query(query, {'node_id': node_id})
        return summary['nodes_deleted'] > 0
    
    def clear_diagram(self, diagram_id: str) -> Dict[str, int]:
        """
        Delete all data for a specific diagram.
        
        Args:
            diagram_id: Diagram ID to clear
            
        Returns:
            Summary of deleted items
        """
        query = """
        MATCH (n {diagram_id: $diagram_id})
        OPTIONAL MATCH (n)-[r {diagram_id: $diagram_id}]-()
        DELETE r, n
        RETURN count(DISTINCT n) as nodes_deleted, count(DISTINCT r) as relationships_deleted
        """
        
        results = self.execute_query(query, {'diagram_id': diagram_id})
        if results:
            return {
                'nodes_deleted': results[0]['nodes_deleted'],
                'relationships_deleted': results[0]['relationships_deleted']
            }
        return {'nodes_deleted': 0, 'relationships_deleted': 0}
    
    def get_diagram_stats(self, diagram_id: str) -> Dict[str, Any]:
        """
        Get statistics for a diagram.
        
        Args:
            diagram_id: Diagram ID
            
        Returns:
            Diagram statistics
        """
        query = """
        MATCH (n {diagram_id: $diagram_id})
        OPTIONAL MATCH (n)-[r {diagram_id: $diagram_id}]-()
        RETURN 
            count(DISTINCT n) as node_count,
            count(DISTINCT r) as relationship_count,
            collect(DISTINCT labels(n)) as node_types,
            collect(DISTINCT type(r)) as relationship_types
        """
        
        results = self.execute_query(query, {'diagram_id': diagram_id})
        if results:
            result = results[0]
            return {
                'diagram_id': diagram_id,
                'node_count': result['node_count'],
                'relationship_count': result['relationship_count'],
                'node_types': [t for types in result['node_types'] for t in types if t],
                'relationship_types': [t for t in result['relationship_types'] if t]
            }
        
        return {
            'diagram_id': diagram_id,
            'node_count': 0,
            'relationship_count': 0,
            'node_types': [],
            'relationship_types': []
        }
    
    @abstractmethod
    def create(self, entity: T) -> T:
        """Create a new entity."""
        pass
    
    @abstractmethod
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass
    
    @abstractmethod
    def list_by_diagram(self, diagram_id: str, limit: int = 100) -> List[T]:
        """List entities for a diagram."""
        pass
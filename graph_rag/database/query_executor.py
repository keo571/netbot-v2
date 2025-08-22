"""
Query execution infrastructure for Neo4j operations.
"""

import time
from typing import List, Dict, Optional

from models.graph_models import GraphNode, GraphRelationship


class QueryExecutor:
    """Handles query execution with timing and profiling"""
    
    def __init__(self, connection=None):
        self.connection = connection
    
    def execute_with_timing(self, session, query: str, params: dict = None, 
                           operation_name: str = "Query", profile: bool = False):
        """Execute query with timing and optional profiling"""
        start_time = time.time()
        
        if profile:
            profile_query = f"PROFILE {query}"
            profile_result = session.run(profile_query, params or {})
            print("🔍 Query Profile:")
            print(profile_result.consume().profile)
        
        result = session.run(query, params or {})
        execution_time = time.time() - start_time
        print(f"⏱️    {operation_name}: {execution_time:.2f}s")
        return result


class ResultParser:
    """Parses Neo4j results into domain objects"""
    
    @staticmethod
    def create_node_from_value(value) -> Optional[GraphNode]:
        """Create GraphNode from Neo4j node value"""
        node_id = value.get('id', str(value.element_id))
        
        # Extract properties and remove redundant fields
        props = dict(value)
        label = props.pop('label', '')
        props.pop('id', None)
        diagram_id = props.pop('diagram_id', '')  # Extract diagram_id from properties
        
        return GraphNode(
            id=str(node_id),
            label=label,
            type=list(value.labels)[0] if value.labels else 'Node',
            diagram_id=diagram_id,
            properties=props
        )
    
    @staticmethod
    def create_relationship_from_value(value) -> Optional[GraphRelationship]:
        """Create GraphRelationship from Neo4j relationship value"""
        rel_id = value.get('id', str(value.element_id))
        start_id = value.start_node.get('id', str(value.start_node.element_id))
        end_id = value.end_node.get('id', str(value.end_node.element_id))
        
        # Extract properties and remove redundant fields
        props = dict(value)
        props.pop('id', None)
        diagram_id = props.pop('diagram_id', '')  # Extract diagram_id from properties
        
        return GraphRelationship(
            id=str(rel_id),
            source_id=str(start_id),
            target_id=str(end_id),
            type=value.type,
            diagram_id=diagram_id,
            properties=props
        )
    
    @staticmethod
    def parse_query_results(result) -> Dict[str, List]:
        """Parse Neo4j result into nodes and relationships"""
        nodes = []
        relationships = []
        node_ids = set()
        relationship_ids = set()
        
        for record in result:
            for value in record.values():
                if value is None:
                    continue
                
                if hasattr(value, 'labels'):
                    node = ResultParser.create_node_from_value(value)
                    if node and node.id not in node_ids:
                        node_ids.add(node.id)
                        nodes.append(node)
                
                elif hasattr(value, 'type'):
                    rel = ResultParser.create_relationship_from_value(value)
                    if rel and rel.id not in relationship_ids:
                        relationship_ids.add(rel.id)
                        relationships.append(rel)
        
        print(f"✅ Parsed {len(nodes)} nodes and {len(relationships)} relationships")
        return {"nodes": nodes, "relationships": relationships}
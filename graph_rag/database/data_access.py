"""
Pure database access operations for Neo4j.
Used by both vector search and Cypher query approaches.
"""

from typing import List, Optional
import numpy as np

from models.graph_models import GraphNode, GraphRelationship


class DataAccess:
    """Handles pure database operations - used by both vector and Cypher approaches"""
    
    def __init__(self, connection):
        self.connection = connection
    
    def get_all_nodes(self, diagram_id: str) -> List[GraphNode]:
        """Get all nodes for a diagram (with or without embeddings)"""
        try:
            with self.connection.get_session() as session:
                query = """
                MATCH (n) WHERE n.diagram_id = $diagram_id
                RETURN n.id as id, n.label as label, labels(n)[0] as type, 
                       properties(n) as properties, n.embedding as embedding
                """
                result = session.run(query, diagram_id=diagram_id)
                
                nodes = []
                for record in result:
                    # Convert embedding from list back to numpy array
                    embedding = None
                    if record['embedding']:
                        embedding = np.array(record['embedding'])
                    
                    # Extract properties and remove redundant fields
                    props = dict(record['properties'])
                    props.pop('label', None)     # Remove since it's a separate field
                    props.pop('id', None)        # Remove since it's a separate field
                    props.pop('diagram_id', None) # Remove since it's now a separate field
                    
                    node = GraphNode(
                        id=record['id'],
                        label=record['label'] or '',
                        type=record['type'] or 'Node',
                        diagram_id=diagram_id,
                        properties=props,
                        embedding=embedding
                    )
                    nodes.append(node)
                
                return nodes
                
        except Exception as e:
            print(f"❌ Error getting nodes with embeddings: {e}")
            return []
    
    
    def get_connecting_relationships(self, node_ids: List[str], diagram_id: str) -> List[GraphRelationship]:
        """Get relationships that connect the given nodes"""
        try:
            with self.connection.get_session() as session:
                query = """
                MATCH (n)-[r]->(m) 
                WHERE n.id IN $node_ids AND m.id IN $node_ids 
                AND n.diagram_id = $diagram_id AND m.diagram_id = $diagram_id
                RETURN r.id as id, n.id as source_id, m.id as target_id, type(r) as type,
                       properties(r) as properties
                """
                result = session.run(query, node_ids=node_ids, diagram_id=diagram_id)
                
                relationships = []
                for record in result:
                    # Extract properties and remove redundant fields
                    props = dict(record['properties'])
                    props.pop('diagram_id', None) # Remove since it's now a separate field
                    
                    rel = GraphRelationship(
                        id=record['id'],
                        source_id=record['source_id'],
                        target_id=record['target_id'],
                        type=record['type'],
                        diagram_id=diagram_id,
                        properties=props
                    )
                    relationships.append(rel)
                
                return relationships
                
        except Exception as e:
            print(f"❌ Error getting connecting relationships: {e}")
            return []
    
    def get_paths_between_nodes(self, node_ids: List[str], diagram_id: str, max_hops: int = 3) -> List[GraphRelationship]:
        """Get all relationships in paths between relevant nodes (including intermediate nodes)"""
        try:
            with self.connection.get_session() as session:
                query = f"""
                MATCH path = (start)-[*1..{max_hops}]->(end)
                WHERE start.id IN $node_ids AND end.id IN $node_ids 
                AND start.diagram_id = $diagram_id AND end.diagram_id = $diagram_id
                AND start.id <> end.id
                UNWIND relationships(path) as r
                RETURN DISTINCT r.id as id, startNode(r).id as source_id, endNode(r).id as target_id, 
                       type(r) as type, properties(r) as properties
                """
                result = session.run(query, node_ids=node_ids, diagram_id=diagram_id)
                
                relationships = []
                seen_rel_ids = set()
                for record in result:
                    rel_id = record['id']
                    if rel_id not in seen_rel_ids:
                        rel = GraphRelationship(
                            id=rel_id,
                            source_id=record['source_id'],
                            target_id=record['target_id'],
                            type=record['type'],
                            diagram_id=diagram_id,
                            properties=dict(record['properties'])
                        )
                        relationships.append(rel)
                        seen_rel_ids.add(rel_id)
                
                return relationships
                
        except Exception as e:
            print(f"❌ Error getting paths between nodes: {e}")
            return []
    
    def get_intermediate_nodes_in_paths(self, node_ids: List[str], diagram_id: str, max_hops: int = 3) -> List[GraphNode]:
        """Get intermediate nodes that lie on paths between relevant nodes"""
        try:
            with self.connection.get_session() as session:
                query = f"""
                MATCH path = (start)-[*1..{max_hops}]->(end)
                WHERE start.id IN $node_ids AND end.id IN $node_ids 
                AND start.diagram_id = $diagram_id AND end.diagram_id = $diagram_id
                AND start.id <> end.id
                UNWIND nodes(path) as n
                WITH n
                WHERE NOT n.id IN $node_ids
                RETURN DISTINCT n.id as id, n.label as label, labels(n)[0] as type, 
                       properties(n) as properties
                """
                result = session.run(query, node_ids=node_ids, diagram_id=diagram_id)
                
                nodes = []
                for record in result:
                    props = dict(record['properties'])
                    label = props.pop('label', '')
                    props.pop('id', None)
                    props.pop('diagram_id', None)  # Remove since it's now a separate field
                    
                    node = GraphNode(
                        id=record['id'],
                        label=label,
                        type=record['type'] or 'Node',
                        diagram_id=diagram_id,
                        properties=props
                    )
                    nodes.append(node)
                
                return nodes
                
        except Exception as e:
            print(f"❌ Error getting intermediate nodes: {e}")
            return []
    
    def get_node_by_id(self, node_id: str, diagram_id: str) -> Optional[GraphNode]:
        """Get a specific node by ID"""
        try:
            with self.connection.get_session() as session:
                query = """
                MATCH (n) WHERE n.id = $node_id AND n.diagram_id = $diagram_id
                RETURN n.id as id, n.label as label, labels(n)[0] as type, 
                       properties(n) as properties
                """
                result = session.run(query, node_id=node_id, diagram_id=diagram_id)
                record = result.single()
                
                if record:
                    # Extract properties and remove redundant fields
                    props = dict(record['properties'])
                    props.pop('label', None)     # Remove since it's a separate field
                    props.pop('id', None)        # Remove since it's a separate field
                    props.pop('diagram_id', None) # Remove since it's now a separate field
                    
                    return GraphNode(
                        id=record['id'],
                        label=record['label'] or '',
                        type=record['type'] or 'Node',
                        diagram_id=diagram_id,
                        properties=props
                    )
                
                return None
                
        except Exception as e:
            print(f"❌ Error getting node by ID: {e}")
            return None
    
    def run_diagnostic_queries(self, diagram_id: str) -> str:
        """Run diagnostic queries to help understand why no data was found"""
        try:
            with self.connection.get_session() as session:
                # Check if diagram_id exists
                result = session.run("MATCH (n) WHERE n.diagram_id = $diagram_id RETURN count(n) as node_count", 
                                   diagram_id=diagram_id)
                node_count = result.single()["node_count"]
                
                if node_count == 0:
                    # Check what diagram_ids actually exist
                    result = session.run("MATCH (n) WHERE n.diagram_id IS NOT NULL RETURN DISTINCT n.diagram_id as diagram_id LIMIT 10")
                    existing_ids = [record["diagram_id"] for record in result]
                    return f"Diagram ID '{diagram_id}' not found. Available diagram IDs: {existing_ids}"
                else:
                    # Show sample data for this diagram_id
                    result = session.run("""
                        MATCH (n) WHERE n.diagram_id = $diagram_id 
                        RETURN n.label as label, labels(n) as types, keys(n) as properties 
                        LIMIT 5
                    """, diagram_id=diagram_id)
                    samples = []
                    for record in result:
                        samples.append(f"{record['label']} ({record['types'][0] if record['types'] else 'Unknown'})")
                    return f"Found {node_count} nodes for diagram '{diagram_id}'. Sample nodes: {', '.join(samples)}"
                    
        except Exception as e:
            return f"Diagnostic failed: {e}"
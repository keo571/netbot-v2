"""
Phase 3: CSV generation and Neo4j storage.
"""

import os
import csv
from typing import List, Tuple, Optional, Set
from neo4j import GraphDatabase

from models.graph_models import GraphNode, GraphRelationship


class KnowledgeGraphExporter:
    """Phase 3: Exports graph data to CSV files and stores in Neo4j database"""
    
    def __init__(self, neo4j_uri: Optional[str] = None, 
                 neo4j_user: Optional[str] = None, 
                 neo4j_password: Optional[str] = None):
        """
        Initialize the exporter with optional Neo4j credentials.
        
        Args:
            neo4j_uri: Neo4j database URI (e.g., bolt://localhost:7687)
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None
    
    def connect_neo4j(self) -> bool:
        """Connect to Neo4j database and create performance indexes"""
        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            print("⚠️ Neo4j credentials not provided, skipping connection.")
            return False
            
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            self.driver.verify_connectivity()
            
            # Create performance indexes automatically
            self._create_performance_indexes()
            return True
        except Exception as e:
            print(f"❌ Error connecting to Neo4j: {e}")
            return False
    
    def _create_performance_indexes(self) -> bool:
        """
        Create indexes to optimize both diagram processing and downstream operations.
        
        These indexes are created upfront during graph storage to ensure optimal performance
        for all operations that will be performed on the graph data throughout its lifecycle.
        """
        indexes = [
            # === NODE INDEXES ===
            
            # Primary diagram filtering - optimizes: MATCH (n {diagram_id: $id})
            # Used by: diagram retrieval, diagram-scoped queries, data isolation
            "CREATE INDEX node_diagram_id IF NOT EXISTS FOR (n) ON (n.diagram_id)",
            
            # Bulk node updates - optimizes: MATCH (n {id: $node_id, diagram_id: $diagram_id}) SET n.property = $value
            # Used by: embedding generation, property updates, node modifications
            # Rationale: Composite index allows instant node lookup without scanning all nodes with same ID across diagrams
            "CREATE INDEX node_id_diagram IF NOT EXISTS FOR (n) ON (n.id, n.diagram_id)",
            
            # Embedding workflow support - optimizes: MATCH (n) WHERE n.embedding IS NULL
            # Used by: embedding generation pipelines to find nodes needing vector embeddings
            # Rationale: Enables efficient identification of nodes requiring embedding processing
            "CREATE INDEX node_embedding IF NOT EXISTS FOR (n) ON (n.embedding)",
            
            # Label-based queries - optimizes: MATCH (n) WHERE n.label CONTAINS $text
            # Used by: search, filtering, content-based retrieval
            "CREATE INDEX node_label IF NOT EXISTS FOR (n) ON (n.label)",
            
            # === RELATIONSHIP INDEXES ===
            
            # Primary relationship filtering - optimizes: MATCH ()-[r {diagram_id: $id}]-()
            # Used by: relationship retrieval, diagram-scoped relationship queries
            "CREATE INDEX rel_diagram_id IF NOT EXISTS FOR ()-[r]-() ON (r.diagram_id)",
            
            # Bulk relationship updates - optimizes: MATCH ()-[r {id: $rel_id, diagram_id: $diagram_id}]-() SET r.property = $value
            # Used by: relationship property updates, bulk operations
            # Rationale: Composite index enables instant relationship lookup for bulk operations
            "CREATE INDEX rel_id_diagram IF NOT EXISTS FOR ()-[r]-() ON (r.id, r.diagram_id)"
        ]
        
        try:
            with self.driver.session(database="neo4j") as session:
                for index_query in indexes:
                    try:
                        session.run(index_query)
                    except Exception as e:
                        # Index might already exist, continue with others
                        pass
                
                return True
                
        except Exception as e:
            print(f"⚠️ Warning: Could not create indexes: {e}")
            return False
    
    def generate_csv_files(self, nodes: List[GraphNode], relationships: List[GraphRelationship], 
                          output_dir: str = "output", subfolder_name: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate CSV files for nodes and relationships.
        
        Args:
            nodes: List of graph nodes to export
            relationships: List of graph relationships to export
            output_dir: Base output directory
            subfolder_name: Optional subfolder name (usually diagram_id)
            
        Returns:
            Tuple of (nodes_file_path, relationships_file_path)
        """
        print("Phase 3: Generating CSV files...")
        
        final_output_dir = self._prepare_output_directory(output_dir, subfolder_name)
        
        nodes_file = self._generate_nodes_csv(nodes, final_output_dir)
        relationships_file = self._generate_relationships_csv(relationships, final_output_dir)
        
        return nodes_file, relationships_file
    
    def _prepare_output_directory(self, output_dir: str, subfolder_name: Optional[str]) -> str:
        """Prepare the output directory, creating subfolder if name provided."""
        if subfolder_name:
            final_output_dir = os.path.join(output_dir, subfolder_name)
        else:
            final_output_dir = output_dir
            
        os.makedirs(final_output_dir, exist_ok=True)
        return final_output_dir
    
    def _generate_nodes_csv(self, nodes: List[GraphNode], output_dir: str) -> str:
        """Generate nodes.csv file with dynamic columns based on node properties."""
        nodes_file = os.path.join(output_dir, "nodes.csv")
        
        # Collect all unique properties across nodes
        all_properties = self._collect_all_node_properties(nodes)
        header = ['id', 'label', 'type'] + sorted(all_properties)
        
        with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for node in nodes:
                row = [node.id, node.label, node.type]
                row.extend(node.properties.get(prop_key, '') for prop_key in sorted(all_properties))
                writer.writerow(row)
        
        return nodes_file
    
    def _generate_relationships_csv(self, relationships: List[GraphRelationship], output_dir: str) -> str:
        """Generate relationships.csv file with dynamic columns and consistent IDs."""
        relationships_file = os.path.join(output_dir, "relationships.csv")
        
        # Collect all unique properties across relationships
        all_properties = self._collect_all_relationship_properties(relationships)
        header = ['id', 'source_id', 'target_id', 'type'] + sorted(all_properties)
        
        with open(relationships_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for i, relationship in enumerate(relationships):
                rel_id = f"rel-{i+1}"  # Consistent relationship ID
                row = [rel_id, relationship.source_id, relationship.target_id, relationship.type]
                row.extend(relationship.properties.get(prop_key, '') for prop_key in sorted(all_properties))
                writer.writerow(row)
        
        return relationships_file
    
    def _collect_all_node_properties(self, nodes: List[GraphNode]) -> Set[str]:
        """Collect all unique property keys from all nodes."""
        all_properties: Set[str] = set()
        for node in nodes:
            all_properties.update(node.properties.keys())
        return all_properties
    
    def _collect_all_relationship_properties(self, relationships: List[GraphRelationship]) -> Set[str]:
        """Collect all unique property keys from all relationships."""
        all_properties: Set[str] = set()
        for relationship in relationships:
            all_properties.update(relationship.properties.keys())
        return all_properties
    
    def store_in_neo4j(self, diagram_id: str, nodes: List[GraphNode], relationships: List[GraphRelationship]) -> bool:
        """
        Store nodes and relationships in Neo4j database with diagram partitioning.
        
        Args:
            diagram_id: Unique identifier for this diagram
            nodes: List of nodes to store
            relationships: List of relationships to store
            
        Returns:
            True if storage succeeded, False otherwise
        """
        if not self._ensure_neo4j_connection():
            return False
        
        try:
            success = self._execute_graph_storage(diagram_id, nodes, relationships)
            return success
                
        except Exception as e:
            print(f"❌ Error storing in Neo4j: {e}")
            return False
        finally:
            self.close()
    
    def _diagram_exists(self, diagram_id: str) -> bool:
        """Check if a diagram with the given ID already exists in Neo4j."""
        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(
                    "MATCH (n {diagram_id: $diagram_id}) RETURN count(n) as node_count LIMIT 1",
                    diagram_id=diagram_id
                )
                record = result.single()
                return record["node_count"] > 0 if record else False
        except Exception as e:
            print(f"⚠️ Warning: Could not check for existing diagram: {e}")
            return False
    
    def _get_existing_diagram_data(self, diagram_id: str) -> tuple:
        """Get existing nodes and relationships for a diagram from Neo4j."""
        try:
            with self.driver.session(database="neo4j") as session:
                # Get nodes
                nodes_result = session.run(
                    "MATCH (n {diagram_id: $diagram_id}) RETURN n.id as id, n.label as label, n.type as type, properties(n) as properties",
                    diagram_id=diagram_id
                )
                nodes = []
                for record in nodes_result:
                    node_dict = {
                        'id': record['id'],
                        'label': record['label'],
                        'type': record['type'],
                        'properties': dict(record['properties'])
                    }
                    # Remove diagram_id from properties to avoid duplication
                    node_dict['properties'].pop('diagram_id', None)
                    nodes.append(node_dict)
                
                # Get relationships
                rels_result = session.run(
                    "MATCH (a {diagram_id: $diagram_id})-[r]->(b {diagram_id: $diagram_id}) "
                    "RETURN a.id as source_id, b.id as target_id, type(r) as type, properties(r) as properties",
                    diagram_id=diagram_id
                )
                relationships = []
                for i, record in enumerate(rels_result):
                    rel_dict = {
                        'id': f"rel-{i+1}",
                        'source_id': record['source_id'],
                        'target_id': record['target_id'],
                        'type': record['type'],
                        'properties': dict(record['properties'])
                    }
                    # Remove diagram_id from properties to avoid duplication
                    rel_dict['properties'].pop('diagram_id', None)
                    relationships.append(rel_dict)
                
                return nodes, relationships
        except Exception as e:
            print(f"⚠️ Warning: Could not retrieve existing diagram data: {e}")
            return [], []
    
    def _delete_diagram(self, diagram_id: str) -> bool:
        """Delete all nodes and relationships for a specific diagram from Neo4j."""
        try:
            with self.driver.session(database="neo4j") as session:
                # Delete all relationships first (to avoid constraint issues)
                session.run(
                    "MATCH ()-[r {diagram_id: $diagram_id}]-() DELETE r",
                    diagram_id=diagram_id
                )
                # Then delete all nodes
                session.run(
                    "MATCH (n {diagram_id: $diagram_id}) DELETE n",
                    diagram_id=diagram_id
                )
                print(f"🗑️ Deleted existing diagram '{diagram_id}' from Neo4j")
                return True
        except Exception as e:
            print(f"⚠️ Warning: Could not delete existing diagram: {e}")
            return False
    
    def _ensure_neo4j_connection(self) -> bool:
        """Ensure Neo4j connection is established."""
        return self.connect_neo4j()
    
    def _execute_graph_storage(self, diagram_id: str, nodes: List[GraphNode], relationships: List[GraphRelationship]) -> bool:
        """Execute the actual graph data storage in Neo4j."""
        with self.driver.session(database="neo4j") as session:
            session.execute_write(self._create_nodes_and_relationships, diagram_id, nodes, relationships)
            return True
    
    @staticmethod
    def _create_nodes_and_relationships(tx, diagram_id: str, nodes: List[GraphNode], relationships: List[GraphRelationship]) -> None:
        """Create nodes and relationships in a Neo4j transaction."""
        KnowledgeGraphExporter._create_nodes(tx, diagram_id, nodes)
        KnowledgeGraphExporter._create_relationships(tx, diagram_id, relationships)
    
    @staticmethod
    def _create_nodes(tx, diagram_id: str, nodes: List[GraphNode]) -> None:
        """Create nodes in Neo4j with proper type sanitization."""
        for node in nodes:
            sanitized_type = ''.join(filter(str.isalnum, node.type))
            query = f"""
            MERGE (n:{sanitized_type} {{id: $id, diagram_id: $diagram_id}})
            SET n.label = $label, n.type = $type, n.embedding = $embedding, n += $properties
            """
            # Explicitly set embedding for future embedding workflows
            # Use null - the embedding index will handle finding nodes needing embeddings
            embedding_value = node.embedding if node.embedding is not None else None
            tx.run(query, 
                   id=node.id, 
                   diagram_id=diagram_id, 
                   label=node.label, 
                   type=node.type,
                   embedding=embedding_value,
                   properties=node.properties)
    
    @staticmethod
    def _create_relationships(tx, diagram_id: str, relationships: List[GraphRelationship]) -> None:
        """Create relationships in Neo4j with consistent IDs matching CSV export."""
        for i, relationship in enumerate(relationships):
            rel_id = f"rel-{i+1}"  # Consistent with CSV generation
            
            query = f"""
            MATCH (a {{id: $source_id, diagram_id: $diagram_id}})
            MATCH (b {{id: $target_id, diagram_id: $diagram_id}})
            MERGE (a)-[r:{relationship.type} {{id: $rel_id, diagram_id: $diagram_id}}]->(b)
            SET r += $properties
            """
            tx.run(query,
                   diagram_id=diagram_id,
                   rel_id=rel_id,
                   source_id=relationship.source_id, 
                   target_id=relationship.target_id, 
                   properties=relationship.properties)

    def close(self) -> None:
        """Close Neo4j connection and cleanup resources."""
        if self.driver:
            self.driver.close()
            self.driver = None

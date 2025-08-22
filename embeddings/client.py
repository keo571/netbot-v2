"""
Add embeddings to existing graphs in Neo4j without re-processing images.
"""

from typing import List, Dict

from graph_rag.database.connection import Neo4jConnection
from models.graph_models import GraphNode
from .embedding_encoder import EmbeddingEncoder


class EmbeddingManager:
    """Self-contained embedding manager for adding embeddings to existing graphs"""
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        """
        Initialize EmbeddingManager with Neo4j credentials.
        
        Args:
            neo4j_uri: Neo4j database URI (defaults to env NEO4J_URI)
            neo4j_user: Neo4j username (defaults to env NEO4J_USER)
            neo4j_password: Neo4j password (defaults to env NEO4J_PASSWORD)
        """
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Use provided credentials or fall back to environment variables
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
        
        # Create own connection
        self.connection = Neo4jConnection(self.neo4j_uri, self.neo4j_user, self.neo4j_password)
        self.embedding_encoder = EmbeddingEncoder()
    
    def add_embeddings(self, diagram_id: str, batch_size: int = 100) -> bool:
        """Add embeddings to all nodes in an existing diagram"""
        print(f"🧠 Adding embeddings to existing diagram: {diagram_id}")
        
        # Check if diagram exists
        if not self._diagram_exists(diagram_id):
            print(f"❌ Diagram '{diagram_id}' not found in database")
            return False
        
        # Check if embeddings already exist (all-or-nothing per diagram)
        if self._embeddings_exist(diagram_id):
            print(f"✅ Diagram already has embeddings")
            return True
        
        # Add embeddings to all nodes
        processed = self._process_all_nodes(diagram_id, batch_size)
        print(f"✅ Added embeddings to {processed} nodes")
        return True
    
    def _diagram_exists(self, diagram_id: str) -> bool:
        """Check if diagram exists in database"""
        with self.connection.get_session() as session:
            query = "MATCH (n) WHERE n.diagram_id = $diagram_id RETURN true LIMIT 1"
            result = session.run(query, diagram_id=diagram_id)
            return result.single() is not None
    
    def _embeddings_exist(self, diagram_id: str) -> bool:
        """Check if embeddings exist for diagram (all-or-nothing per diagram)"""
        with self.connection.get_session() as session:
            query = """
            MATCH (n) WHERE n.diagram_id = $diagram_id 
            RETURN n.embedding IS NOT NULL as has_embedding LIMIT 1
            """
            result = session.run(query, diagram_id=diagram_id)
            record = result.single()
            return record["has_embedding"] if record else False
    
    
    def _process_all_nodes(self, diagram_id: str, batch_size: int) -> int:
        """Add embeddings to all nodes in diagram"""
        with self.connection.get_session() as session:
            # Get all nodes (embeddings are all-or-nothing, so all nodes need processing)
            query = """
            MATCH (n) WHERE n.diagram_id = $diagram_id
            RETURN n.id as id, n.label as label, labels(n)[0] as type, properties(n) as properties
            """
            result = session.run(query, diagram_id=diagram_id)
            
            # Collect nodes to process
            nodes_to_process = []
            for record in result:
                props = dict(record['properties'])
                
                node = GraphNode(
                    id=record['id'],
                    label=record['label'] or '',
                    type=record['type'] or 'Node',
                    diagram_id=diagram_id,
                    properties=props
                )
                nodes_to_process.append(node)
            
            if not nodes_to_process:
                return 0
            
            # Process in batches
            for i in range(0, len(nodes_to_process), batch_size):
                batch = nodes_to_process[i:i + batch_size]
                
                # Compute embeddings for batch (much faster than one-by-one)
                embeddings = self.embedding_encoder.batch_compute_embeddings(batch)
                
                batch_updates = []
                for node, embedding in zip(batch, embeddings):
                    batch_updates.append({
                        'node_id': node.id,
                        'embedding': embedding.tolist()
                    })
                
                # Update Neo4j with batch
                update_query = """
                UNWIND $batch as item
                MATCH (n) WHERE n.id = item.node_id AND n.diagram_id = $diagram_id
                SET n.embedding = item.embedding
                """
                session.run(update_query, batch=batch_updates, diagram_id=diagram_id)
                            
            return len(nodes_to_process)
    
    def list_diagrams_with_embeddings(self) -> List[str]:
        """List all diagrams that have embeddings"""
        with self.connection.get_session() as session:
            query = """
            MATCH (n)
            WITH n.diagram_id as diagram_id, collect(n)[0] as sample_node
            WHERE sample_node.embedding IS NOT NULL
            RETURN diagram_id
            ORDER BY diagram_id
            """
            result = session.run(query)
            return [record["diagram_id"] for record in result]
    
    def bulk_add_embeddings(self, diagram_ids: List[str], batch_size: int = 100) -> Dict[str, bool]:
        """
        Add embeddings to multiple diagrams in bulk.
        
        Args:
            diagram_ids: List of diagram IDs to process
            batch_size: Number of nodes to process at once per diagram
            
        Returns:
            Dict mapping diagram_id to success status
        """
        results = {}
        total_diagrams = len(diagram_ids)
        
        print(f"🧠 Starting bulk embedding operation for {total_diagrams} diagrams...")
        
        for i, diagram_id in enumerate(diagram_ids, 1):
            print(f"📋 Processing diagram {i}/{total_diagrams}: {diagram_id}")
            
            try:
                success = self.add_embeddings(diagram_id, batch_size)
                results[diagram_id] = success
                
                if success:
                    print(f"✅ {diagram_id}: Embeddings added successfully")
                else:
                    print(f"⚠️ {diagram_id}: Embeddings already exist or diagram not found")
                    
            except Exception as e:
                print(f"❌ {diagram_id}: Failed - {str(e)}")
                results[diagram_id] = False
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        failed = total_diagrams - successful
        
        print(f"\n📊 Bulk embedding summary:")
        print(f"✅ Successful: {successful}/{total_diagrams}")
        if failed > 0:
            print(f"❌ Failed: {failed}/{total_diagrams}")
            
        return results
    
    def bulk_check_embeddings(self, diagram_ids: List[str]) -> Dict[str, bool]:
        """
        Check embedding status for multiple diagrams.
        
        Args:
            diagram_ids: List of diagram IDs to check
            
        Returns:
            Dict mapping diagram_id to embedding existence status
        """
        results = {}
        
        print(f"🔍 Checking embedding status for {len(diagram_ids)} diagrams...")
        
        for diagram_id in diagram_ids:
            try:
                has_embeddings = self._embeddings_exist(diagram_id)
                results[diagram_id] = has_embeddings
                status = "✅ Has embeddings" if has_embeddings else "❌ No embeddings"
                print(f"  {diagram_id}: {status}")
            except Exception as e:
                print(f"  {diagram_id}: ❌ Error checking - {str(e)}")
                results[diagram_id] = False
        
        return results
    
    def get_diagrams_without_embeddings(self) -> List[str]:
        """
        Get list of diagrams that don't have embeddings yet.
        
        Returns:
            List of diagram IDs without embeddings
        """
        with self.connection.get_session() as session:
            query = """
            MATCH (n) 
            WHERE n.diagram_id IS NOT NULL
            WITH n.diagram_id as diagram_id, collect(n)[0] as sample_node
            WHERE sample_node.embedding IS NULL
            RETURN diagram_id
            ORDER BY diagram_id
            """
            result = session.run(query)
            return [record["diagram_id"] for record in result]
    
    def close(self):
        """Clean up resources."""
        if self.connection:
            self.connection.close()

"""
Neo4j database connection management.
"""

from neo4j import GraphDatabase
from typing import Optional


class Neo4jConnection:
    """Manages Neo4j database connections"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver: Optional[GraphDatabase.driver] = None
    
    def connect(self) -> bool:
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            self.driver.verify_connectivity()
            print("✅ Connected to Neo4j")
            return True
        except Exception as e:
            print(f"❌ Error connecting to Neo4j: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
            print("🔌 Neo4j connection closed")
    
    def is_connected(self) -> bool:
        """Check if connected to Neo4j"""
        return self.driver is not None
    
    def get_session(self):
        """Get Neo4j session (context manager)"""
        if not self.driver:
            if not self.connect():
                raise ConnectionError("Cannot connect to Neo4j")
        return self.driver.session()
    
    def create_performance_indexes(self) -> bool:
        """Create indexes to improve query performance based on actual query patterns"""
        indexes = [
            # Node indexes - optimized for actual query patterns
            "CREATE INDEX node_diagram_id IF NOT EXISTS FOR (n) ON (n.diagram_id)",  # Most frequent filter
            "CREATE INDEX node_id_diagram IF NOT EXISTS FOR (n) ON (n.id, n.diagram_id)",  # Composite index for bulk updates
            "CREATE INDEX node_embedding IF NOT EXISTS FOR (n) ON (n.embedding)",  # Used to find NULL embeddings
            "CREATE INDEX node_label IF NOT EXISTS FOR (n) ON (n.label)",  # Used in some queries
            # Relationship indexes - based on actual relationship queries
            "CREATE INDEX rel_diagram_id IF NOT EXISTS FOR ()-[r]-() ON (r.diagram_id)",  # Filter by diagram
            "CREATE INDEX rel_id_diagram IF NOT EXISTS FOR ()-[r]-() ON (r.id, r.diagram_id)",  # Composite index for bulk updates
            "CREATE INDEX rel_embedding IF NOT EXISTS FOR ()-[r]-() ON (r.embedding)"  # Used to find NULL embeddings
        ]
        
        try:
            with self.get_session() as session:
                for index_query in indexes:
                    try:
                        session.run(index_query)
                        index_name = index_query.split('FOR')[0].split('INDEX')[1].split('IF')[0].strip()
                        print(f"✅ Created index: {index_name}")
                    except Exception as e:
                        print(f"⚠️ Index creation skipped: {e}")
            return True
        except Exception as e:
            print(f"❌ Error creating indexes: {e}")
            return False
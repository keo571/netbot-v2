"""
Neo4j database connection management with connection pooling.
"""

from neo4j import GraphDatabase
from typing import Optional, Dict
import threading


class Neo4jConnectionPool:
    """Singleton connection pool for Neo4j"""
    
    _instance = None
    _lock = threading.Lock()
    _pools: Dict[str, GraphDatabase.driver] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_driver(cls, uri: str, user: str, password: str) -> GraphDatabase.driver:
        """Get or create a pooled driver for the given connection params"""
        pool_key = f"{uri}:{user}"
        
        if pool_key not in cls._pools:
            with cls._lock:
                if pool_key not in cls._pools:
                    try:
                        driver = GraphDatabase.driver(
                            uri,
                            auth=(user, password),
                            max_connection_lifetime=30 * 60,  # 30 minutes
                            max_connection_pool_size=10,
                            connection_acquisition_timeout=60
                        )
                        driver.verify_connectivity()
                        cls._pools[pool_key] = driver
                        print(f"✅ Created Neo4j connection pool for {uri}")
                    except Exception as e:
                        print(f"❌ Error creating Neo4j connection pool: {e}")
                        raise
        
        return cls._pools[pool_key]
    
    @classmethod
    def close_all(cls):
        """Close all connection pools"""
        with cls._lock:
            for pool_key, driver in cls._pools.items():
                try:
                    driver.close()
                    print(f"🔌 Closed Neo4j connection pool: {pool_key}")
                except Exception as e:
                    print(f"⚠️ Error closing connection pool {pool_key}: {e}")
            cls._pools.clear()


class Neo4jConnection:
    """Manages Neo4j database connections using connection pooling"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.pool = Neo4jConnectionPool()
        self._driver: Optional[GraphDatabase.driver] = None
    
    def connect(self) -> bool:
        """Get connection from pool"""
        try:
            self._driver = self.pool.get_driver(self.uri, self.user, self.password)
            return True
        except Exception as e:
            print(f"❌ Error getting connection from pool: {e}")
            return False
    
    def close(self):
        """No-op - connections are managed by the pool"""
        # Don't actually close the driver, it's managed by the pool
        self._driver = None
    
    def is_connected(self) -> bool:
        """Check if we have access to the pooled driver"""
        return self._driver is not None or self.connect()
    
    def get_session(self):
        """Get Neo4j session from pooled connection"""
        if not self._driver:
            if not self.connect():
                raise ConnectionError("Cannot get connection from pool")
        return self._driver.session()
    
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
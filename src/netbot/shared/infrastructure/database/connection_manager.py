"""
Enhanced database connection management with improved pooling and monitoring.

This centralizes all Neo4j connection logic and provides a clean interface
for all services to use.
"""

import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, Optional, Any, Generator
from neo4j import GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from ...config.settings import get_settings
from ...exceptions import DatabaseError, ConfigurationError


class DatabaseManager:
    """
    Centralized database connection manager with enhanced pooling.
    
    Provides thread-safe connection pooling, health monitoring, and
    automatic reconnection capabilities.
    """
    
    _instance: Optional["DatabaseManager"] = None
    _lock = threading.RLock()
    _drivers: Dict[str, GraphDatabase.driver] = {}
    
    def __new__(cls) -> "DatabaseManager":
        """Singleton pattern to ensure single connection manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the database manager (called once due to singleton)."""
        if self._initialized:
            return
            
        self.settings = get_settings()
        self._health_check_interval = 60  # seconds
        self._last_health_check = 0
        self._is_healthy = False
        self._initialized = True
    
    def get_driver(self, connection_name: str = "default") -> GraphDatabase.driver:
        """
        Get or create a database driver with connection pooling.
        
        Args:
            connection_name: Name of the connection (for multiple databases)
        
        Returns:
            Neo4j driver instance
        
        Raises:
            DatabaseError: If connection cannot be established
            ConfigurationError: If database configuration is invalid
        """
        if connection_name not in self._drivers:
            with self._lock:
                if connection_name not in self._drivers:
                    self._drivers[connection_name] = self._create_driver()
        
        # Health check if needed
        self._check_health_if_needed()
        
        if not self._is_healthy:
            raise DatabaseError("Database is not healthy")
        
        return self._drivers[connection_name]
    
    def _create_driver(self) -> GraphDatabase.driver:
        """Create a new Neo4j driver with optimized settings."""
        try:
            config = self.settings.database_config
            
            driver = GraphDatabase.driver(
                config['uri'],
                auth=(config['user'], config['password']),
                
                # Connection pool settings
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=config['max_connections'],
                connection_acquisition_timeout=config['connection_timeout'],
                
                # Performance optimizations
                keep_alive=True,
                max_retry_time=30,
                initial_retry_delay=1,
                retry_delay_multiplier=2,
                
                # Monitoring
                connection_timeout=10,
                encrypted=False,  # Set to True for production
            )
            
            # Verify connectivity
            driver.verify_connectivity()
            
            print(f"✅ Created Neo4j driver for {config['uri']}")
            return driver
            
        except AuthError as e:
            raise ConfigurationError(f"Database authentication failed: {e}")
        except ServiceUnavailable as e:
            raise DatabaseError(f"Database service unavailable: {e}")
        except Exception as e:
            raise DatabaseError(f"Failed to create database driver: {e}")
    
    def _check_health_if_needed(self):
        """Perform health check if enough time has passed."""
        now = time.time()
        if now - self._last_health_check > self._health_check_interval:
            self._health_check()
            self._last_health_check = now
    
    def _health_check(self):
        """Check database health."""
        try:
            for driver_name, driver in self._drivers.items():
                with driver.session() as session:
                    result = session.run("RETURN 1 as health_check")
                    result.single()
            
            self._is_healthy = True
            
        except Exception as e:
            print(f"⚠️ Database health check failed: {e}")
            self._is_healthy = False
    
    @contextmanager
    def get_session(self, connection_name: str = "default") -> Generator[Session, None, None]:
        """
        Get a database session with proper resource management.
        
        Args:
            connection_name: Name of the connection
            
        Yields:
            Neo4j session
        """
        driver = self.get_driver(connection_name)
        session = driver.session()
        try:
            yield session
        except Exception as e:
            # Log error but don't suppress it
            print(f"❌ Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            session.close()
    
    def execute_query(self, 
                     query: str, 
                     parameters: Dict[str, Any] = None,
                     connection_name: str = "default") -> Any:
        """
        Execute a query with proper error handling.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            connection_name: Database connection to use
            
        Returns:
            Query results
        """
        with self.get_session(connection_name) as session:
            try:
                result = session.run(query, parameters or {})
                return list(result)  # Consume all results
            except Exception as e:
                raise DatabaseError(f"Query execution failed: {e}")
    
    def execute_write_transaction(self,
                                 transaction_func,
                                 *args,
                                 connection_name: str = "default",
                                 **kwargs):
        """
        Execute a write transaction with retry logic.
        
        Args:
            transaction_func: Function to execute in transaction
            connection_name: Database connection to use
            *args, **kwargs: Arguments for transaction function
            
        Returns:
            Transaction result
        """
        driver = self.get_driver(connection_name)
        with driver.session() as session:
            try:
                return session.execute_write(transaction_func, *args, **kwargs)
            except Exception as e:
                raise DatabaseError(f"Write transaction failed: {e}")
    
    def execute_read_transaction(self,
                                transaction_func,
                                *args,
                                connection_name: str = "default", 
                                **kwargs):
        """
        Execute a read transaction with retry logic.
        
        Args:
            transaction_func: Function to execute in transaction
            connection_name: Database connection to use
            *args, **kwargs: Arguments for transaction function
            
        Returns:
            Transaction result
        """
        driver = self.get_driver(connection_name)
        with driver.session() as session:
            try:
                return session.execute_read(transaction_func, *args, **kwargs)
            except Exception as e:
                raise DatabaseError(f"Read transaction failed: {e}")
    
    def close_all(self):
        """Close all database connections."""
        with self._lock:
            for connection_name, driver in self._drivers.items():
                try:
                    driver.close()
                    print(f"🔌 Closed database connection: {connection_name}")
                except Exception as e:
                    print(f"⚠️ Error closing connection {connection_name}: {e}")
            
            self._drivers.clear()
            self._is_healthy = False
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        stats = {}
        for name, driver in self._drivers.items():
            # This is a simplified version - Neo4j driver doesn't expose
            # detailed pool stats easily
            stats[name] = {
                'is_healthy': self._is_healthy,
                'last_health_check': self._last_health_check,
            }
        return stats


# Global instance
_db_manager = None
_db_lock = threading.Lock()


@lru_cache()
def get_database() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        DatabaseManager singleton instance
    """
    global _db_manager
    
    if _db_manager is None:
        with _db_lock:
            if _db_manager is None:
                _db_manager = DatabaseManager()
    
    return _db_manager


# Cleanup function for application shutdown
def close_database_connections():
    """Close all database connections on application shutdown."""
    global _db_manager
    if _db_manager:
        _db_manager.close_all()
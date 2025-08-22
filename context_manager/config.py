"""
Configuration management for the Context Manager system.

Handles environment variables, default settings, and configuration validation
for all storage backends and system components.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Redis configuration for session storage."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    prefix: str = "session:"
    max_connections: int = 20
    socket_timeout: int = 30
    
    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Create Redis config from environment variables."""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            prefix=os.getenv("REDIS_SESSION_PREFIX", "session:"),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "30"))
        )
    
    def to_connection_string(self) -> str:
        """Generate Redis connection string."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class MongoConfig:
    """MongoDB configuration for conversation history."""
    connection_string: str = "mongodb://localhost:27017/"
    database_name: str = "context_manager"
    collection_name: str = "conversation_history"
    max_pool_size: int = 50
    server_selection_timeout_ms: int = 30000
    
    @classmethod
    def from_env(cls) -> "MongoConfig":
        """Create MongoDB config from environment variables."""
        # Build connection string from components or use full string
        connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        
        if not connection_string:
            host = os.getenv("MONGODB_HOST", "localhost")
            port = os.getenv("MONGODB_PORT", "27017")
            username = os.getenv("MONGODB_USERNAME")
            password = os.getenv("MONGODB_PASSWORD")
            
            if username and password:
                connection_string = f"mongodb://{username}:{password}@{host}:{port}/"
            else:
                connection_string = f"mongodb://{host}:{port}/"
        
        return cls(
            connection_string=connection_string,
            database_name=os.getenv("MONGODB_DATABASE", "context_manager"),
            collection_name=os.getenv("MONGODB_COLLECTION", "conversation_history"),
            max_pool_size=int(os.getenv("MONGODB_MAX_POOL_SIZE", "50")),
            server_selection_timeout_ms=int(os.getenv("MONGODB_TIMEOUT_MS", "30000"))
        )


@dataclass
class PostgreSQLConfig:
    """PostgreSQL configuration for user preferences."""
    host: str = "localhost"
    port: int = 5432
    database: str = "context_manager"
    username: str = "postgres"
    password: str = ""
    table_name: str = "user_preferences"
    max_connections: int = 20
    connection_timeout: int = 30
    
    @classmethod
    def from_env(cls) -> "PostgreSQLConfig":
        """Create PostgreSQL config from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DATABASE", "context_manager"),
            username=os.getenv("POSTGRES_USERNAME", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            table_name=os.getenv("POSTGRES_TABLE", "user_preferences"),
            max_connections=int(os.getenv("POSTGRES_MAX_CONNECTIONS", "20")),
            connection_timeout=int(os.getenv("POSTGRES_TIMEOUT", "30"))
        )
    
    def to_connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class ContextManagerConfig:
    """Main Context Manager configuration."""
    session_timeout_seconds: int = 1800  # 30 minutes
    max_conversation_history: int = 1000  # Max exchanges to keep
    max_active_entities: int = 10
    max_diagram_ids: int = 5
    enable_implicit_learning: bool = True
    min_retrieval_score: float = 0.4
    max_retrieval_results: int = 10
    enable_result_diversification: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "ContextManagerConfig":
        """Create config from environment variables."""
        return cls(
            session_timeout_seconds=int(os.getenv("CM_SESSION_TIMEOUT", "1800")),
            max_conversation_history=int(os.getenv("CM_MAX_HISTORY", "1000")),
            max_active_entities=int(os.getenv("CM_MAX_ENTITIES", "10")),
            max_diagram_ids=int(os.getenv("CM_MAX_DIAGRAM_IDS", "5")),
            enable_implicit_learning=os.getenv("CM_ENABLE_LEARNING", "true").lower() == "true",
            min_retrieval_score=float(os.getenv("CM_MIN_SCORE", "0.4")),
            max_retrieval_results=int(os.getenv("CM_MAX_RESULTS", "10")),
            enable_result_diversification=os.getenv("CM_ENABLE_DIVERSIFICATION", "true").lower() == "true",
            log_level=os.getenv("CM_LOG_LEVEL", "INFO")
        )


@dataclass
class StorageConfig:
    """Complete storage configuration."""
    redis: RedisConfig = field(default_factory=RedisConfig)
    mongodb: MongoConfig = field(default_factory=MongoConfig)
    postgresql: PostgreSQLConfig = field(default_factory=PostgreSQLConfig)
    use_in_memory: bool = False  # For testing/development
    
    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create storage config from environment variables."""
        return cls(
            redis=RedisConfig.from_env(),
            mongodb=MongoConfig.from_env(),
            postgresql=PostgreSQLConfig.from_env(),
            use_in_memory=os.getenv("CM_USE_IN_MEMORY", "false").lower() == "true"
        )


@dataclass
class Config:
    """Complete system configuration."""
    context_manager: ContextManagerConfig = field(default_factory=ContextManagerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create complete config from environment variables."""
        return cls(
            context_manager=ContextManagerConfig.from_env(),
            storage=StorageConfig.from_env()
        )
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return any errors.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate basic constraints
        if self.context_manager.session_timeout_seconds < 60:
            errors.append("Session timeout must be at least 60 seconds")
        
        if self.context_manager.max_conversation_history < 10:
            errors.append("Max conversation history must be at least 10")
        
        if self.context_manager.min_retrieval_score < 0 or self.context_manager.min_retrieval_score > 1:
            errors.append("Min retrieval score must be between 0 and 1")
        
        # Validate storage configs if not using in-memory
        if not self.storage.use_in_memory:
            # Redis validation
            if not self.storage.redis.host:
                errors.append("Redis host is required")
            
            if self.storage.redis.port < 1 or self.storage.redis.port > 65535:
                errors.append("Redis port must be between 1 and 65535")
            
            # MongoDB validation
            if not self.storage.mongodb.connection_string:
                errors.append("MongoDB connection string is required")
            
            if not self.storage.mongodb.database_name:
                errors.append("MongoDB database name is required")
            
            # PostgreSQL validation
            if not self.storage.postgresql.host:
                errors.append("PostgreSQL host is required")
            
            if not self.storage.postgresql.username:
                errors.append("PostgreSQL username is required")
            
            if not self.storage.postgresql.database:
                errors.append("PostgreSQL database is required")
        
        return errors
    
    def setup_logging(self):
        """Setup logging based on configuration."""
        logging.basicConfig(
            level=getattr(logging, self.context_manager.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def load_config(config_file: Optional[str] = None) -> Config:
    """
    Load configuration from environment variables and optional config file.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Complete configuration object
    """
    # Load from environment
    config = Config.from_env()
    
    # Load from config file if provided
    if config_file and Path(config_file).exists():
        try:
            import json
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Override environment config with file config
            # This is a simple implementation - could be more sophisticated
            if 'session_timeout_seconds' in file_config:
                config.context_manager.session_timeout_seconds = file_config['session_timeout_seconds']
            
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        raise ValueError(error_msg)
    
    # Setup logging
    config.setup_logging()
    
    logger.info("Configuration loaded and validated successfully")
    return config


def create_example_env_file(output_path: str = ".env.example"):
    """
    Create an example environment file with all configuration options.
    
    Args:
        output_path: Path to write the example file
    """
    env_content = """# Context Manager Configuration

# Redis Configuration (Session Storage)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_SESSION_PREFIX=session:
REDIS_MAX_CONNECTIONS=20
REDIS_SOCKET_TIMEOUT=30

# MongoDB Configuration (Conversation History)
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/
# OR use individual components:
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USERNAME=
MONGODB_PASSWORD=
MONGODB_DATABASE=context_manager
MONGODB_COLLECTION=conversation_history
MONGODB_MAX_POOL_SIZE=50
MONGODB_TIMEOUT_MS=30000

# PostgreSQL Configuration (User Preferences)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=context_manager
POSTGRES_USERNAME=postgres
POSTGRES_PASSWORD=
POSTGRES_TABLE=user_preferences
POSTGRES_MAX_CONNECTIONS=20
POSTGRES_TIMEOUT=30

# Context Manager Settings
CM_SESSION_TIMEOUT=1800  # Session timeout in seconds (30 minutes)
CM_MAX_HISTORY=1000      # Maximum conversation exchanges to keep
CM_MAX_ENTITIES=10       # Maximum active entities to track
CM_MAX_DIAGRAM_IDS=5     # Maximum diagram IDs to remember
CM_ENABLE_LEARNING=true  # Enable implicit preference learning
CM_MIN_SCORE=0.4         # Minimum retrieval score threshold
CM_MAX_RESULTS=10        # Maximum retrieval results to process
CM_ENABLE_DIVERSIFICATION=true  # Enable result diversification
CM_LOG_LEVEL=INFO        # Logging level (DEBUG, INFO, WARNING, ERROR)

# Development/Testing
CM_USE_IN_MEMORY=false   # Use in-memory storage instead of external databases
"""
    
    try:
        with open(output_path, 'w') as f:
            f.write(env_content.strip())
        
        print(f"Created example environment file: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create example env file: {e}")


# Default configuration instance
default_config = Config()
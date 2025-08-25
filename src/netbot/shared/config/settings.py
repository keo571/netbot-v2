"""
Centralized configuration management for NetBot V2.

All environment variables and settings are managed here to ensure consistency
across all services and eliminate configuration duplication.
"""

import os
from functools import lru_cache
from typing import Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """
    Centralized settings for NetBot V2.
    
    All configuration is loaded from environment variables with sensible defaults.
    Uses Pydantic for validation and type safety.
    """
    
    # === Application Settings ===
    app_name: str = Field(default="NetBot V2", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # === API Settings ===
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # === Security Settings ===
    jwt_secret_key: str = Field(default="your-secret-key-change-this-in-production", description="JWT secret key", validation_alias="JWT_SECRET_KEY")
    admin_api_key: str = Field(default="admin-secret-key-change-this", description="Admin API key", validation_alias="ADMIN_API_KEY")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiration time")
    
    # === Database Settings ===
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j database URI", validation_alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username", validation_alias="NEO4J_USER")
    neo4j_password: Optional[str] = Field(default=None, description="Neo4j password", validation_alias="NEO4J_PASSWORD")
    neo4j_max_connections: int = Field(default=10, description="Max Neo4j connections")
    neo4j_connection_timeout: int = Field(default=60, description="Neo4j connection timeout")
    
    # === Redis Settings (for caching) ===
    redis_url: Optional[str] = Field(default=None, description="Redis URL for caching", validation_alias="REDIS_URL")
    redis_ttl: int = Field(default=3600, description="Default Redis TTL in seconds")
    
    # === AI Service Settings ===
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key", validation_alias="GEMINI_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key", validation_alias="OPENAI_API_KEY")
    default_llm_model: str = Field(default="gemini-2.5-flash", description="Default LLM model")
    
    # === Google Cloud Settings ===
    google_application_credentials: Optional[str] = Field(default=None, description="Path to Google credentials JSON", validation_alias="GOOGLE_APPLICATION_CREDENTIALS")
    
    # === Embedding Settings ===
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    embedding_cache_size: int = Field(default=1000, description="Embedding cache size")
    embedding_batch_size: int = Field(default=100, description="Embedding batch size")
    
    # === File Storage Settings ===
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    temp_dir: Path = Field(default=Path("temp_uploads"), description="Temporary uploads directory")
    max_file_size: int = Field(default=50 * 1024 * 1024, description="Max file size in bytes (50MB)")
    
    # === Processing Settings ===
    max_concurrent_processing: int = Field(default=5, description="Max concurrent diagram processing")
    processing_timeout: int = Field(default=300, description="Processing timeout in seconds")
    
    # === Search Settings ===
    default_search_top_k: int = Field(default=8, description="Default search top-k results")
    vector_search_timeout: int = Field(default=30, description="Vector search timeout")
    
    # === Monitoring Settings ===
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics endpoint port")
    
    # === Logging Configuration ===
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            'level': self.log_level,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    
    # === Monitoring Configuration ===
    @property
    def monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return {
            'enabled': self.enable_metrics,
            'port': self.metrics_port,
            'max_history': 1000
        }
    
    # === Cache Configuration ===
    @property
    def cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return {
            'redis_url': self.redis_url,
            'ttl': self.redis_ttl,
            'memory_size': 1000
        }
    
    # === Embedding Configuration ===
    @property
    def embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return {
            'model': self.embedding_model_name,
            'cache_size': self.embedding_cache_size,
            'batch_size': self.embedding_batch_size
        }
    
    @field_validator('neo4j_password', mode='before')
    @classmethod
    def validate_neo4j_password(cls, v):
        if v is None:
            raise ValueError("NEO4J_PASSWORD is required")
        return v
    
    @field_validator('data_dir', 'temp_dir', mode='before')
    @classmethod
    def validate_paths(cls, v):
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration as dictionary."""
        return {
            'uri': self.neo4j_uri,
            'user': self.neo4j_user,
            'password': self.neo4j_password,
            'max_connections': self.neo4j_max_connections,
            'connection_timeout': self.neo4j_connection_timeout,
        }
    
    @property
    def ai_config(self) -> Dict[str, Any]:
        """Get AI service configuration as dictionary."""
        return {
            'gemini_api_key': self.gemini_api_key,
            'openai_api_key': self.openai_api_key,
            'default_model': self.default_llm_model,
            'google_credentials': self.google_application_credentials,
        }
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug and self.jwt_secret_key != "your-secret-key-change-this-in-production"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to ensure settings are only loaded once per application lifecycle.
    """
    return Settings()
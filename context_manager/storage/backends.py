"""
Storage interfaces for the Context Manager system.

Provides implementations for different storage backends:
- Redis for session state (fast, volatile)
- MongoDB for conversation history (persistent, document-based)
- PostgreSQL for user preferences (persistent, relational)
"""

import json
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

# Redis imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# MongoDB imports
try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# PostgreSQL imports  
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    from ..models import SessionState, ConversationHistory, UserPreferences
except ImportError:
    from models import SessionState, ConversationHistory, UserPreferences


logger = logging.getLogger(__name__)


# Abstract base classes for storage interfaces

class SessionStore(ABC):
    """Abstract interface for session state storage."""
    
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieve session state by ID."""
        pass
    
    @abstractmethod
    def save_session(self, session: SessionState, ttl_seconds: int = 1800) -> bool:
        """Save session state with TTL."""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete session state."""
        pass
    
    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        pass


class HistoryStore(ABC):
    """Abstract interface for conversation history storage."""
    
    @abstractmethod
    def get_history(self, user_id: str) -> Optional[ConversationHistory]:
        """Retrieve conversation history for user."""
        pass
    
    @abstractmethod
    def save_history(self, history: ConversationHistory) -> bool:
        """Save conversation history."""
        pass
    
    @abstractmethod
    def delete_history(self, user_id: str) -> bool:
        """Delete conversation history for user."""
        pass


class UserStore(ABC):
    """Abstract interface for user preferences storage."""
    
    @abstractmethod
    def get_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Retrieve user preferences."""
        pass
    
    @abstractmethod
    def save_preferences(self, preferences: UserPreferences) -> bool:
        """Save user preferences."""
        pass
    
    @abstractmethod
    def delete_preferences(self, user_id: str) -> bool:
        """Delete user preferences."""
        pass


# Redis implementation for session storage

class RedisSessionStore(SessionStore):
    """Redis-based session storage for fast access."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 password: Optional[str] = None, prefix: str = "session:"):
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for RedisSessionStore")
        self.client = redis.Redis(host=host, port=port, db=db, password=password, 
                                decode_responses=True)
        self.prefix = prefix
        
        # Test connection
        try:
            self.client.ping()
            logger.info("Connected to Redis successfully")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _get_key(self, session_id: str) -> str:
        """Get Redis key for session."""
        return f"{self.prefix}{session_id}"
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieve session from Redis."""
        try:
            key = self._get_key(session_id)
            data = self.client.get(key)
            if data:
                session_dict = json.loads(data)
                return SessionState.from_dict(session_dict)
            return None
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    def save_session(self, session: SessionState, ttl_seconds: int = 1800) -> bool:
        """Save session to Redis with TTL."""
        try:
            key = self._get_key(session.session_id)
            data = json.dumps(session.to_dict())
            self.client.setex(key, ttl_seconds, data)
            return True
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session from Redis."""
        try:
            key = self._get_key(session_id)
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def exists(self, session_id: str) -> bool:
        """Check if session exists in Redis."""
        try:
            key = self._get_key(session_id)
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking session {session_id}: {e}")
            return False
    
    def extend_ttl(self, session_id: str, ttl_seconds: int = 1800) -> bool:
        """Extend session TTL."""
        try:
            key = self._get_key(session_id)
            return bool(self.client.expire(key, ttl_seconds))
        except Exception as e:
            logger.error(f"Error extending TTL for session {session_id}: {e}")
            return False


# MongoDB implementation for conversation history

class MongoHistoryStore(HistoryStore):
    """MongoDB-based conversation history storage."""
    
    def __init__(self, connection_string: str, database_name: str = "context_manager",
                 collection_name: str = "conversation_history"):
        """Initialize MongoDB connection."""
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo is required for MongoDB support")
        
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection: Collection = self.db[collection_name]
        
        # Create index on user_id for efficient queries
        self.collection.create_index("user_id", unique=True)
        
        # Test connection
        try:
            self.client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def get_history(self, user_id: str) -> Optional[ConversationHistory]:
        """Retrieve conversation history from MongoDB."""
        try:
            doc = self.collection.find_one({"user_id": user_id})
            if doc:
                # Remove MongoDB's _id field
                doc.pop('_id', None)
                return ConversationHistory.from_dict(doc)
            return None
        except Exception as e:
            logger.error(f"Error retrieving history for user {user_id}: {e}")
            return None
    
    def save_history(self, history: ConversationHistory) -> bool:
        """Save conversation history to MongoDB."""
        try:
            doc = history.to_dict()
            self.collection.replace_one(
                {"user_id": history.user_id},
                doc,
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error saving history for user {history.user_id}: {e}")
            return False
    
    def delete_history(self, user_id: str) -> bool:
        """Delete conversation history from MongoDB."""
        try:
            result = self.collection.delete_one({"user_id": user_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting history for user {user_id}: {e}")
            return False
    
    def get_histories_by_timeframe(self, hours: int = 24) -> List[ConversationHistory]:
        """Get all histories updated in the last N hours."""
        try:
            cutoff = datetime.now() - timedelta(hours=hours)
            cursor = self.collection.find({"updated_at": {"$gte": cutoff.isoformat()}})
            
            histories = []
            for doc in cursor:
                doc.pop('_id', None)
                histories.append(ConversationHistory.from_dict(doc))
            
            return histories
        except Exception as e:
            logger.error(f"Error retrieving recent histories: {e}")
            return []


# PostgreSQL implementation for user preferences

class PostgreSQLUserStore(UserStore):
    """PostgreSQL-based user preferences storage."""
    
    def __init__(self, connection_string: str, table_name: str = "user_preferences"):
        """Initialize PostgreSQL connection."""
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL support")
        
        self.connection_string = connection_string
        self.table_name = table_name
        
        # Test connection and create table if needed
        try:
            self._create_table()
            logger.info("Connected to PostgreSQL successfully")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.connection_string)
    
    def _create_table(self):
        """Create user preferences table if it doesn't exist."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        user_id VARCHAR(255) PRIMARY KEY,
                        response_style VARCHAR(50) DEFAULT 'balanced',
                        expertise_level VARCHAR(50) DEFAULT 'intermediate',
                        preferred_formats JSONB DEFAULT '["mermaid_diagram", "bullet_points", "paragraph"]',
                        topic_interest_profile JSONB DEFAULT '[]',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
    
    def get_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Retrieve user preferences from PostgreSQL."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(
                        f"SELECT * FROM {self.table_name} WHERE user_id = %s",
                        (user_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        # Convert to dictionary and create UserPreferences
                        data = dict(row)
                        return UserPreferences.from_dict(data)
                    return None
        except Exception as e:
            logger.error(f"Error retrieving preferences for user {user_id}: {e}")
            return None
    
    def save_preferences(self, preferences: UserPreferences) -> bool:
        """Save user preferences to PostgreSQL."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    data = preferences.to_dict()
                    cursor.execute(f"""
                        INSERT INTO {self.table_name} 
                        (user_id, response_style, expertise_level, preferred_formats, 
                         topic_interest_profile, created_at, updated_at)
                        VALUES (%(user_id)s, %(response_style)s, %(expertise_level)s, 
                               %(preferred_formats)s, %(topic_interest_profile)s, 
                               %(created_at)s, %(updated_at)s)
                        ON CONFLICT (user_id) DO UPDATE SET
                            response_style = EXCLUDED.response_style,
                            expertise_level = EXCLUDED.expertise_level,
                            preferred_formats = EXCLUDED.preferred_formats,
                            topic_interest_profile = EXCLUDED.topic_interest_profile,
                            updated_at = EXCLUDED.updated_at
                    """, data)
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error saving preferences for user {preferences.user_id}: {e}")
            return False
    
    def delete_preferences(self, user_id: str) -> bool:
        """Delete user preferences from PostgreSQL."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"DELETE FROM {self.table_name} WHERE user_id = %s",
                        (user_id,)
                    )
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting preferences for user {user_id}: {e}")
            return False


# In-memory implementations for testing/development

class InMemorySessionStore(SessionStore):
    """In-memory session storage for testing."""
    
    def __init__(self):
        self._sessions = {}
        self._expiry = {}
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieve session from memory."""
        # Check if expired
        if session_id in self._expiry and datetime.now() > self._expiry[session_id]:
            self.delete_session(session_id)
            return None
        
        return self._sessions.get(session_id)
    
    def save_session(self, session: SessionState, ttl_seconds: int = 1800) -> bool:
        """Save session to memory with TTL."""
        self._sessions[session.session_id] = session
        self._expiry[session.session_id] = datetime.now() + timedelta(seconds=ttl_seconds)
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session from memory."""
        existed = session_id in self._sessions
        self._sessions.pop(session_id, None)
        self._expiry.pop(session_id, None)
        return existed
    
    def exists(self, session_id: str) -> bool:
        """Check if session exists in memory."""
        return self.get_session(session_id) is not None


class InMemoryHistoryStore(HistoryStore):
    """In-memory history storage for testing."""
    
    def __init__(self):
        self._histories = {}
    
    def get_history(self, user_id: str) -> Optional[ConversationHistory]:
        """Retrieve history from memory."""
        return self._histories.get(user_id)
    
    def save_history(self, history: ConversationHistory) -> bool:
        """Save history to memory."""
        self._histories[history.user_id] = history
        return True
    
    def delete_history(self, user_id: str) -> bool:
        """Delete history from memory."""
        existed = user_id in self._histories
        self._histories.pop(user_id, None)
        return existed


class InMemoryUserStore(UserStore):
    """In-memory user preferences storage for testing."""
    
    def __init__(self):
        self._preferences = {}
    
    def get_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Retrieve preferences from memory."""
        return self._preferences.get(user_id)
    
    def save_preferences(self, preferences: UserPreferences) -> bool:
        """Save preferences to memory."""
        self._preferences[preferences.user_id] = preferences
        return True
    
    def delete_preferences(self, user_id: str) -> bool:
        """Delete preferences from memory."""
        existed = user_id in self._preferences
        self._preferences.pop(user_id, None)
        return existed
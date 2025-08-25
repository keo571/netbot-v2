"""
Pluggable storage backends for Context Manager.

Provides different storage options for sessions, messages, and users
depending on deployment needs and persistence requirements.
"""

import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ....shared import get_logger, get_cache_manager
from ..models import Session, Message, User, SessionStatus


class StorageBackend(ABC):
    """Abstract base class for Context Manager storage backends."""
    
    @abstractmethod
    def store_session(self, session: Session) -> None:
        """Store a session."""
        pass
    
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        pass
    
    @abstractmethod
    def update_session(self, session: Session) -> None:
        """Update an existing session."""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        pass
    
    @abstractmethod
    def store_message(self, message: Message) -> None:
        """Store a message."""
        pass
    
    @abstractmethod
    def get_session_messages(self, session_id: str, limit: int = 100) -> List[Message]:
        """Get messages for a session."""
        pass
    
    @abstractmethod
    def store_user(self, user: User) -> None:
        """Store a user."""
        pass
    
    @abstractmethod
    def get_user(self, user_id: str) -> Optional[User]:
        """Retrieve a user by ID."""
        pass
    
    @abstractmethod
    def update_user(self, user: User) -> None:
        """Update an existing user."""
        pass
    
    @abstractmethod
    def cleanup_expired_sessions(self, timeout_minutes: int = 30) -> int:
        """Clean up expired sessions."""
        pass


class InMemoryStorage(StorageBackend):
    """
    In-memory storage backend for development and testing.
    
    Provides fast, ephemeral storage that doesn't persist between restarts.
    Good for development, testing, and stateless deployments.
    """
    
    def __init__(self):
        """Initialize in-memory storage."""
        self.logger = get_logger(__name__)
        self._lock = threading.RLock()
        
        # In-memory data stores
        self._sessions: Dict[str, Session] = {}
        self._messages: Dict[str, List[Message]] = {}  # session_id -> messages
        self._users: Dict[str, User] = {}
        
        self.logger.info("Initialized InMemoryStorage backend")
    
    def store_session(self, session: Session) -> None:
        """Store a session in memory."""
        with self._lock:
            self._sessions[session.session_id] = session
            self.logger.debug(f"Stored session: {session.session_id}")
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session from memory."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                # Check if session should be expired
                if session.should_expire():
                    session.status = SessionStatus.EXPIRED
                    self._sessions[session_id] = session
            return session
    
    def update_session(self, session: Session) -> None:
        """Update a session in memory."""
        with self._lock:
            if session.session_id in self._sessions:
                self._sessions[session.session_id] = session
                self.logger.debug(f"Updated session: {session.session_id}")
            else:
                self.logger.warning(f"Attempted to update non-existent session: {session.session_id}")
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session from memory."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                # Also delete associated messages
                if session_id in self._messages:
                    del self._messages[session_id]
                self.logger.debug(f"Deleted session: {session_id}")
    
    def store_message(self, message: Message) -> None:
        """Store a message in memory."""
        with self._lock:
            if message.session_id not in self._messages:
                self._messages[message.session_id] = []
            
            self._messages[message.session_id].append(message)
            self.logger.debug(f"Stored message: {message.message_id} for session: {message.session_id}")
    
    def get_session_messages(self, session_id: str, limit: int = 100) -> List[Message]:
        """Get messages for a session from memory."""
        with self._lock:
            messages = self._messages.get(session_id, [])
            # Return messages sorted by timestamp, limited
            sorted_messages = sorted(messages, key=lambda m: m.created_at)
            return sorted_messages[-limit:] if len(sorted_messages) > limit else sorted_messages
    
    def store_user(self, user: User) -> None:
        """Store a user in memory."""
        with self._lock:
            self._users[user.user_id] = user
            self.logger.debug(f"Stored user: {user.user_id}")
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Retrieve a user from memory."""
        with self._lock:
            return self._users.get(user_id)
    
    def update_user(self, user: User) -> None:
        """Update a user in memory."""
        with self._lock:
            self._users[user.user_id] = user
            self.logger.debug(f"Updated user: {user.user_id}")
    
    def cleanup_expired_sessions(self, timeout_minutes: int = 30) -> int:
        """Clean up expired sessions from memory."""
        with self._lock:
            expired_sessions = []
            
            for session_id, session in self._sessions.items():
                if session.should_expire(timeout_minutes):
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                self.delete_session(session_id)
            
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            return len(expired_sessions)
    
    def get_active_sessions(self, user_id: str) -> List[Session]:
        """Get active sessions for a user."""
        with self._lock:
            active_sessions = []
            for session in self._sessions.values():
                if (session.user_id == user_id and 
                    session.status == SessionStatus.ACTIVE and 
                    not session.should_expire()):
                    active_sessions.append(session)
            
            return sorted(active_sessions, key=lambda s: s.last_activity, reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            total_messages = sum(len(msgs) for msgs in self._messages.values())
            active_sessions = sum(1 for s in self._sessions.values() if s.status == SessionStatus.ACTIVE)
            
            return {
                'backend': 'memory',
                'sessions': len(self._sessions),
                'active_sessions': active_sessions,
                'users': len(self._users),
                'total_messages': total_messages,
                'memory_usage': 'not_tracked'
            }


class ExternalStorage(StorageBackend):
    """
    External storage backend using the shared database infrastructure.
    
    Provides persistent storage with proper database backing for production use.
    Uses the Context Manager repository for actual database operations.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize external storage backend.
        
        Args:
            database_url: Database connection URL (optional, uses shared config)
        """
        self.logger = get_logger(__name__)
        self.cache = get_cache_manager()
        
        # Import repository here to avoid circular imports
        from ..repository import ContextRepository
        self.repository = ContextRepository()
        
        # Cache settings
        self.cache_ttl = 300  # 5 minutes
        self.cache_namespace = 'context_storage'
        
        self.logger.info("Initialized ExternalStorage backend with database persistence")
    
    def store_session(self, session: Session) -> None:
        """Store a session in the database."""
        try:
            self.repository.create_session(session)
            
            # Cache the session for fast access
            cache_key = f"session_{session.session_id}"
            self.cache.set(cache_key, session, namespace=self.cache_namespace, ttl_seconds=self.cache_ttl)
            
            self.logger.debug(f"Stored session in database: {session.session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store session {session.session_id}: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session from cache or database."""
        try:
            # Check cache first
            cache_key = f"session_{session_id}"
            cached_session = self.cache.get(cache_key, namespace=self.cache_namespace)
            
            if cached_session:
                self.logger.debug(f"Retrieved session from cache: {session_id}")
                return cached_session
            
            # Fetch from database
            session = self.repository.get_session(session_id)
            
            if session:
                # Cache for future access
                self.cache.set(cache_key, session, namespace=self.cache_namespace, ttl_seconds=self.cache_ttl)
                self.logger.debug(f"Retrieved session from database: {session_id}")
            
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def update_session(self, session: Session) -> None:
        """Update a session in the database."""
        try:
            self.repository.update_session(session)
            
            # Update cache
            cache_key = f"session_{session.session_id}"
            self.cache.set(cache_key, session, namespace=self.cache_namespace, ttl_seconds=self.cache_ttl)
            
            self.logger.debug(f"Updated session in database: {session.session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update session {session.session_id}: {e}")
            raise
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session from the database."""
        try:
            # Remove from cache
            cache_key = f"session_{session_id}"
            self.cache.delete(cache_key, namespace=self.cache_namespace)
            
            # Note: We don't actually delete from database, just mark as archived
            session = self.get_session(session_id)
            if session:
                session.status = SessionStatus.ARCHIVED
                self.repository.update_session(session)
                
            self.logger.debug(f"Archived session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")
            raise
    
    def store_message(self, message: Message) -> None:
        """Store a message in the database."""
        try:
            self.repository.create_message(message)
            
            # Invalidate session messages cache
            cache_key = f"messages_{message.session_id}"
            self.cache.delete(cache_key, namespace=self.cache_namespace)
            
            self.logger.debug(f"Stored message in database: {message.message_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store message {message.message_id}: {e}")
            raise
    
    def get_session_messages(self, session_id: str, limit: int = 100) -> List[Message]:
        """Get messages for a session from cache or database."""
        try:
            # Check cache first
            cache_key = f"messages_{session_id}_{limit}"
            cached_messages = self.cache.get(cache_key, namespace=self.cache_namespace)
            
            if cached_messages:
                self.logger.debug(f"Retrieved messages from cache for session: {session_id}")
                return cached_messages
            
            # Fetch from database
            messages = self.repository.get_session_messages(session_id, limit)
            
            # Cache the results
            self.cache.set(
                cache_key, 
                messages, 
                namespace=self.cache_namespace, 
                ttl_seconds=self.cache_ttl // 2  # Shorter TTL for messages
            )
            
            self.logger.debug(f"Retrieved {len(messages)} messages from database for session: {session_id}")
            return messages
            
        except Exception as e:
            self.logger.error(f"Failed to get messages for session {session_id}: {e}")
            return []
    
    def store_user(self, user: User) -> None:
        """Store a user in the database."""
        try:
            existing_user = self.repository.get_user(user.user_id)
            
            if existing_user:
                self.repository.update_user(user)
            else:
                self.repository.create_user(user)
            
            # Cache the user
            cache_key = f"user_{user.user_id}"
            self.cache.set(cache_key, user, namespace=self.cache_namespace, ttl_seconds=self.cache_ttl)
            
            self.logger.debug(f"Stored user in database: {user.user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store user {user.user_id}: {e}")
            raise
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Retrieve a user from cache or database."""
        try:
            # Check cache first
            cache_key = f"user_{user_id}"
            cached_user = self.cache.get(cache_key, namespace=self.cache_namespace)
            
            if cached_user:
                self.logger.debug(f"Retrieved user from cache: {user_id}")
                return cached_user
            
            # Fetch from database
            user = self.repository.get_user(user_id)
            
            if user:
                # Cache for future access
                self.cache.set(cache_key, user, namespace=self.cache_namespace, ttl_seconds=self.cache_ttl)
                self.logger.debug(f"Retrieved user from database: {user_id}")
            
            return user
            
        except Exception as e:
            self.logger.error(f"Failed to get user {user_id}: {e}")
            return None
    
    def update_user(self, user: User) -> None:
        """Update a user in the database."""
        try:
            self.repository.update_user(user)
            
            # Update cache
            cache_key = f"user_{user.user_id}"
            self.cache.set(cache_key, user, namespace=self.cache_namespace, ttl_seconds=self.cache_ttl)
            
            self.logger.debug(f"Updated user in database: {user.user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update user {user.user_id}: {e}")
            raise
    
    def cleanup_expired_sessions(self, timeout_minutes: int = 30) -> int:
        """Clean up expired sessions in the database."""
        try:
            expired_count = self.repository.expire_inactive_sessions(timeout_minutes)
            
            # Clear relevant caches
            self.cache.clear_namespace(self.cache_namespace)
            
            self.logger.info(f"Cleaned up {expired_count} expired sessions from database")
            return expired_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def get_active_sessions(self, user_id: str) -> List[Session]:
        """Get active sessions for a user from the database."""
        try:
            return self.repository.get_active_sessions(user_id)
        except Exception as e:
            self.logger.error(f"Failed to get active sessions for user {user_id}: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            # Get some basic stats from repository
            # This is a simplified version - could be expanded
            cache_stats = self.cache.get_stats()
            
            return {
                'backend': 'external_database',
                'cache_stats': cache_stats,
                'cache_namespace': self.cache_namespace,
                'cache_ttl_seconds': self.cache_ttl
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            return {
                'backend': 'external_database',
                'error': str(e)
            }
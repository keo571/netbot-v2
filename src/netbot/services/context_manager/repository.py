"""
Repository layer for Context Manager data persistence.

Handles storage and retrieval of sessions, messages, users, and conversation history.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json

from ...shared import BaseRepository, DatabaseError
from .models import Session, Message, User, ConversationSummary, SessionStatus


class ContextRepository(BaseRepository):
    """
    Repository for Context Manager database operations.
    
    Provides specialized queries for session management, user tracking,
    and conversation history operations.
    """
    
    def create(self, entity) -> Any:
        """Required by base repository - delegates to specific create methods."""
        if isinstance(entity, Session):
            return self.create_session(entity)
        elif isinstance(entity, Message):
            return self.create_message(entity)
        elif isinstance(entity, User):
            return self.create_user(entity)
        else:
            raise ValueError(f"Unsupported entity type: {type(entity)}")
    
    def get_by_id(self, entity_id: str) -> Optional[Any]:
        """Required by base repository - delegates based on ID format."""
        if entity_id.startswith('session_'):
            return self.get_session(entity_id)
        elif entity_id.startswith('user_'):
            return self.get_user(entity_id)
        elif entity_id.startswith('msg_'):
            return self.get_message(entity_id)
        else:
            return None
    
    def list_by_diagram(self, diagram_id: str, limit: int = 100) -> List[Session]:
        """List sessions associated with a diagram."""
        return self.get_sessions_by_diagram(diagram_id, limit)
    
    def update(self, entity) -> Any:
        """Required by base repository - delegates to specific update methods."""
        if isinstance(entity, Session):
            return self.update_session(entity)
        elif isinstance(entity, User):
            return self.update_user(entity)
        else:
            raise ValueError(f"Unsupported entity type for update: {type(entity)}")
    
    # Session operations
    def create_session(self, session: Session) -> Session:
        """Create a new session in the database."""
        query = """
        CREATE (s:Session $properties)
        RETURN s
        """
        
        try:
            properties = self._session_to_db(session)
            self.execute_write_query(query, {'properties': properties})
            return session
            
        except Exception as e:
            raise DatabaseError(f"Failed to create session {session.session_id}: {e}")
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        query = """
        MATCH (s:Session {session_id: $session_id})
        RETURN s
        """
        
        try:
            results = self.execute_query(query, {'session_id': session_id})
            if not results:
                return None
            
            return self._session_from_db(results[0]['s'])
            
        except Exception as e:
            raise DatabaseError(f"Failed to get session {session_id}: {e}")
    
    def update_session(self, session: Session) -> Session:
        """Update an existing session."""
        query = """
        MATCH (s:Session {session_id: $session_id})
        SET s += $properties
        RETURN s
        """
        
        try:
            properties = self._session_to_db(session)
            self.execute_write_query(query, {
                'session_id': session.session_id,
                'properties': properties
            })
            return session
            
        except Exception as e:
            raise DatabaseError(f"Failed to update session {session.session_id}: {e}")
    
    def get_active_sessions(self, user_id: str) -> List[Session]:
        """Get active sessions for a user."""
        query = """
        MATCH (s:Session {user_id: $user_id, status: $status})
        RETURN s
        ORDER BY s.last_activity DESC
        """
        
        try:
            results = self.execute_query(query, {
                'user_id': user_id,
                'status': SessionStatus.ACTIVE.value
            })
            
            return [self._session_from_db(result['s']) for result in results]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get active sessions for user {user_id}: {e}")
    
    def get_sessions_by_diagram(self, diagram_id: str, limit: int = 50) -> List[Session]:
        """Get sessions associated with a specific diagram."""
        query = """
        MATCH (s:Session {diagram_id: $diagram_id})
        RETURN s
        ORDER BY s.last_activity DESC
        LIMIT $limit
        """
        
        try:
            results = self.execute_query(query, {
                'diagram_id': diagram_id,
                'limit': limit
            })
            
            return [self._session_from_db(result['s']) for result in results]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get sessions for diagram {diagram_id}: {e}")
    
    def expire_inactive_sessions(self, timeout_minutes: int = 30) -> int:
        """Expire sessions that have been inactive too long."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        
        query = """
        MATCH (s:Session {status: $active_status})
        WHERE datetime(s.last_activity) < datetime($cutoff_time)
        SET s.status = $expired_status
        RETURN count(s) as expired_count
        """
        
        try:
            results = self.execute_query(query, {
                'active_status': SessionStatus.ACTIVE.value,
                'expired_status': SessionStatus.EXPIRED.value,
                'cutoff_time': cutoff_time.isoformat()
            })
            
            return results[0]['expired_count'] if results else 0
            
        except Exception as e:
            raise DatabaseError(f"Failed to expire inactive sessions: {e}")
    
    # Message operations
    def create_message(self, message: Message) -> Message:
        """Create a new message in the database."""
        query = """
        CREATE (m:Message $properties)
        RETURN m
        """
        
        try:
            properties = self._message_to_db(message)
            self.execute_write_query(query, {'properties': properties})
            return message
            
        except Exception as e:
            raise DatabaseError(f"Failed to create message {message.message_id}: {e}")
    
    def get_message(self, message_id: str) -> Optional[Message]:
        """Get message by ID."""
        query = """
        MATCH (m:Message {message_id: $message_id})
        RETURN m
        """
        
        try:
            results = self.execute_query(query, {'message_id': message_id})
            if not results:
                return None
            
            return self._message_from_db(results[0]['m'])
            
        except Exception as e:
            raise DatabaseError(f"Failed to get message {message_id}: {e}")
    
    def get_session_messages(self, session_id: str, limit: int = 100) -> List[Message]:
        """Get messages for a session, ordered by timestamp."""
        query = """
        MATCH (m:Message {session_id: $session_id})
        RETURN m
        ORDER BY m.created_at ASC
        LIMIT $limit
        """
        
        try:
            results = self.execute_query(query, {
                'session_id': session_id,
                'limit': limit
            })
            
            return [self._message_from_db(result['m']) for result in results]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get messages for session {session_id}: {e}")
    
    def get_recent_messages(self, session_id: str, count: int = 5) -> List[Message]:
        """Get recent messages from a session."""
        query = """
        MATCH (m:Message {session_id: $session_id})
        RETURN m
        ORDER BY m.created_at DESC
        LIMIT $count
        """
        
        try:
            results = self.execute_query(query, {
                'session_id': session_id,
                'count': count
            })
            
            # Return in chronological order (oldest first)
            messages = [self._message_from_db(result['m']) for result in results]
            return list(reversed(messages))
            
        except Exception as e:
            raise DatabaseError(f"Failed to get recent messages for session {session_id}: {e}")
    
    # User operations
    def create_user(self, user: User) -> User:
        """Create a new user in the database."""
        query = """
        CREATE (u:User $properties)
        RETURN u
        """
        
        try:
            properties = self._user_to_db(user)
            self.execute_write_query(query, {'properties': properties})
            return user
            
        except Exception as e:
            raise DatabaseError(f"Failed to create user {user.user_id}: {e}")
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        query = """
        MATCH (u:User {user_id: $user_id})
        RETURN u
        """
        
        try:
            results = self.execute_query(query, {'user_id': user_id})
            if not results:
                return None
            
            return self._user_from_db(results[0]['u'])
            
        except Exception as e:
            raise DatabaseError(f"Failed to get user {user_id}: {e}")
    
    def update_user(self, user: User) -> User:
        """Update an existing user."""
        query = """
        MATCH (u:User {user_id: $user_id})
        SET u += $properties
        RETURN u
        """
        
        try:
            properties = self._user_to_db(user)
            self.execute_write_query(query, {
                'user_id': user.user_id,
                'properties': properties
            })
            return user
            
        except Exception as e:
            raise DatabaseError(f"Failed to update user {user.user_id}: {e}")
    
    def get_user_session_history(self, user_id: str, limit: int = 50) -> List[Session]:
        """Get session history for a user."""
        query = """
        MATCH (s:Session {user_id: $user_id})
        RETURN s
        ORDER BY s.created_at DESC
        LIMIT $limit
        """
        
        try:
            results = self.execute_query(query, {
                'user_id': user_id,
                'limit': limit
            })
            
            return [self._session_from_db(result['s']) for result in results]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get session history for user {user_id}: {e}")
    
    # Analytics and insights
    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a session."""
        query = """
        MATCH (s:Session {session_id: $session_id})
        OPTIONAL MATCH (m:Message {session_id: $session_id})
        RETURN s,
               count(m) as message_count,
               collect(m.message_type) as message_types,
               collect(m.entities_mentioned) as all_entities
        """
        
        try:
            results = self.execute_query(query, {'session_id': session_id})
            if not results:
                return {}
            
            result = results[0]
            session = self._session_from_db(result['s'])
            
            # Process entities
            all_entities = []
            for entity_list in result['all_entities']:
                if entity_list:
                    all_entities.extend(entity_list)
            
            # Count entity frequencies
            entity_counts = {}
            for entity in all_entities:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
            
            return {
                'session_id': session_id,
                'message_count': result['message_count'],
                'duration_minutes': session.duration_minutes,
                'active_entities': list(session.active_entities),
                'entity_frequencies': entity_counts,
                'message_types': list(set(result['message_types'])),
                'last_activity': session.last_activity,
                'status': session.status
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get session statistics: {e}")
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get user behavior insights and patterns."""
        query = """
        MATCH (u:User {user_id: $user_id})
        OPTIONAL MATCH (s:Session {user_id: $user_id})
        OPTIONAL MATCH (m:Message {session_id: s.session_id})
        RETURN u,
               count(DISTINCT s) as session_count,
               count(m) as total_messages,
               collect(DISTINCT s.diagram_id) as diagrams_used,
               collect(m.entities_mentioned) as all_entities,
               avg(duration.inSeconds(datetime(s.created_at), datetime(s.last_activity))/60.0) as avg_session_duration
        """
        
        try:
            results = self.execute_query(query, {'user_id': user_id})
            if not results:
                return {}
            
            result = results[0]
            user = self._user_from_db(result['u']) if result['u'] else None
            
            if not user:
                return {}
            
            # Process entities
            all_entities = []
            for entity_list in result['all_entities']:
                if entity_list:
                    all_entities.extend(entity_list)
            
            return {
                'user_id': user_id,
                'session_count': result['session_count'],
                'total_messages': result['total_messages'],
                'avg_session_duration': result.get('avg_session_duration', 0),
                'diagrams_used': [d for d in result['diagrams_used'] if d],
                'expertise_level': user.expertise_level,
                'response_style': user.response_style,
                'top_entities': user.get_top_entities(),
                'topic_interests': user.topic_interests,
                'last_seen': user.last_seen
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get user insights: {e}")
    
    # Data conversion methods
    def _session_to_db(self, session: Session) -> Dict[str, Any]:
        """Convert Session object to database format."""
        return {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'diagram_id': session.diagram_id,
            'status': session.status.value,
            'active_entities': list(session.active_entities),
            'last_query_intent': session.last_query_intent,
            'conversation_topic': session.conversation_topic,
            'last_activity': session.last_activity.isoformat(),
            'message_count': session.message_count,
            'created_at': session.created_at.isoformat(),
            'metadata': json.dumps(session.metadata),
            'preferences_override': json.dumps(session.preferences_override)
        }
    
    def _session_from_db(self, data: Dict[str, Any]) -> Session:
        """Convert database data to Session object."""
        return Session(
            session_id=data['session_id'],
            user_id=data['user_id'],
            diagram_id=data.get('diagram_id'),
            status=SessionStatus(data.get('status', SessionStatus.ACTIVE.value)),
            active_entities=set(data.get('active_entities', [])),
            last_query_intent=data.get('last_query_intent'),
            conversation_topic=data.get('conversation_topic'),
            last_activity=datetime.fromisoformat(data['last_activity']),
            message_count=data.get('message_count', 0),
            metadata=json.loads(data.get('metadata', '{}')),
            preferences_override=json.loads(data.get('preferences_override', '{}')),
            created_at=datetime.fromisoformat(data['created_at'])
        )
    
    def _message_to_db(self, message: Message) -> Dict[str, Any]:
        """Convert Message object to database format."""
        return {
            'message_id': message.message_id,
            'session_id': message.session_id,
            'content': message.content,
            'message_type': message.message_type.value,
            'metadata': json.dumps(message.metadata),
            'diagram_context': message.diagram_context,
            'entities_mentioned': message.entities_mentioned,
            'intent': message.intent,
            'confidence_score': message.confidence_score,
            'relevance_score': message.relevance_score,
            'created_at': message.created_at.isoformat()
        }
    
    def _message_from_db(self, data: Dict[str, Any]) -> Message:
        """Convert database data to Message object."""
        return Message(
            message_id=data['message_id'],
            session_id=data['session_id'],
            content=data['content'],
            message_type=data['message_type'],
            metadata=json.loads(data.get('metadata', '{}')),
            diagram_context=data.get('diagram_context'),
            entities_mentioned=data.get('entities_mentioned', []),
            intent=data.get('intent'),
            confidence_score=data.get('confidence_score', 1.0),
            relevance_score=data.get('relevance_score', 0.0),
            created_at=datetime.fromisoformat(data['created_at'])
        )
    
    def _user_to_db(self, user: User) -> Dict[str, Any]:
        """Convert User object to database format."""
        return {
            'user_id': user.user_id,
            'preferences': json.dumps(user.preferences),
            'response_style': user.response_style.value,
            'expertise_level': user.expertise_level.value,
            'topic_interests': user.topic_interests,
            'frequent_entities': json.dumps(user.frequent_entities),
            'learned_preferences': json.dumps(user.learned_preferences),
            'total_sessions': user.total_sessions,
            'total_messages': user.total_messages,
            'last_seen': user.last_seen.isoformat(),
            'metadata': json.dumps(user.metadata),
            'created_at': user.created_at.isoformat()
        }
    
    def _user_from_db(self, data: Dict[str, Any]) -> User:
        """Convert database data to User object."""
        from .models import ResponseStyle, ExpertiseLevel
        
        return User(
            user_id=data['user_id'],
            preferences=json.loads(data.get('preferences', '{}')),
            response_style=ResponseStyle(data.get('response_style', ResponseStyle.DETAILED.value)),
            expertise_level=ExpertiseLevel(data.get('expertise_level', ExpertiseLevel.INTERMEDIATE.value)),
            topic_interests=data.get('topic_interests', []),
            frequent_entities=json.loads(data.get('frequent_entities', '{}')),
            learned_preferences=json.loads(data.get('learned_preferences', '{}')),
            total_sessions=data.get('total_sessions', 0),
            total_messages=data.get('total_messages', 0),
            last_seen=datetime.fromisoformat(data['last_seen']),
            metadata=json.loads(data.get('metadata', '{}')),
            created_at=datetime.fromisoformat(data['created_at'])
        )
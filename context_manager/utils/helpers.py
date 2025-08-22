"""
Utility functions for context management, maintenance, and analytics.

Provides helper functions for session cleanup, data analysis, migration,
and system maintenance tasks.
"""

import logging
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import statistics

from ..models import (
    SessionState, ConversationHistory, UserPreferences,
    ConversationExchange, QueryIntent, UserFeedback
)
from ..storage import SessionStore, HistoryStore, UserStore


logger = logging.getLogger(__name__)


class ContextAnalytics:
    """Analytics and reporting for context manager data."""
    
    def __init__(self,
                 session_store: SessionStore,
                 history_store: HistoryStore,
                 user_store: UserStore):
        """Initialize with storage backends."""
        self.session_store = session_store
        self.history_store = history_store
        self.user_store = user_store
    
    def get_active_sessions_count(self) -> int:
        """
        Get count of currently active sessions.
        
        Returns:
            Number of active sessions
        """
        # This would need to be implemented differently for each storage backend
        # For now, return a placeholder
        try:
            # Implementation would depend on storage backend capabilities
            logger.info("Active sessions count requested")
            return 0  # Placeholder
        except Exception as e:
            logger.error(f"Error getting active sessions count: {e}")
            return 0
    
    def get_user_engagement_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get engagement statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with engagement metrics
        """
        try:
            history = self.history_store.get_history(user_id)
            if not history:
                return {"error": "No history found"}
            
            exchanges = history.conversation_log
            if not exchanges:
                return {"total_exchanges": 0}
            
            # Basic metrics
            total_exchanges = len(exchanges)
            
            # Time-based metrics
            first_interaction = min(ex.timestamp for ex in exchanges)
            last_interaction = max(ex.timestamp for ex in exchanges)
            total_duration = (last_interaction - first_interaction).total_seconds()
            
            # Query intent distribution
            intent_counts = {}
            feedback_counts = {"thumb_up": 0, "thumb_down": 0, "none": 0}
            
            for exchange in exchanges:
                # Count feedback
                feedback_counts[exchange.user_feedback.value] += 1
            
            # Calculate satisfaction rate
            total_feedback = feedback_counts["thumb_up"] + feedback_counts["thumb_down"]
            satisfaction_rate = (
                feedback_counts["thumb_up"] / total_feedback 
                if total_feedback > 0 else 0
            )
            
            # Recent activity
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_exchanges = [
                ex for ex in exchanges 
                if ex.timestamp >= recent_cutoff
            ]
            
            return {
                "total_exchanges": total_exchanges,
                "first_interaction": first_interaction.isoformat(),
                "last_interaction": last_interaction.isoformat(),
                "total_duration_hours": total_duration / 3600,
                "satisfaction_rate": satisfaction_rate,
                "feedback_distribution": feedback_counts,
                "recent_exchanges_7d": len(recent_exchanges),
                "avg_exchanges_per_day": total_exchanges / max(1, (total_duration / 86400))
            }
            
        except Exception as e:
            logger.error(f"Error getting engagement stats for user {user_id}: {e}")
            return {"error": str(e)}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get overall system metrics.
        
        Returns:
            System-wide metrics dictionary
        """
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "active_sessions": self.get_active_sessions_count(),
                # Add more system-wide metrics here
                "status": "healthy"
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}


class ContextMaintenance:
    """Maintenance tasks for context manager."""
    
    def __init__(self,
                 session_store: SessionStore,
                 history_store: HistoryStore,
                 user_store: UserStore):
        """Initialize with storage backends."""
        self.session_store = session_store
        self.history_store = history_store
        self.user_store = user_store
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions (implementation depends on storage backend).
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            # This would be implemented differently for each storage backend
            # Redis sessions expire automatically, but we might want to track this
            logger.info("Session cleanup requested")
            return 0  # Placeholder
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            return 0
    
    def archive_old_conversations(self, 
                                days_old: int = 365,
                                archive_path: Optional[str] = None) -> int:
        """
        Archive old conversations to reduce database size.
        
        Args:
            days_old: Archive conversations older than this many days
            archive_path: Optional path to save archived conversations
            
        Returns:
            Number of conversations archived
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            archived_count = 0
            
            # Implementation would depend on storage backend's ability to list users
            # For now, this is a placeholder
            logger.info(f"Would archive conversations older than {cutoff_date}")
            
            return archived_count
            
        except Exception as e:
            logger.error(f"Error archiving old conversations: {e}")
            return 0
    
    def optimize_user_preferences(self, user_id: str) -> bool:
        """
        Optimize user preferences based on usage patterns.
        
        Args:
            user_id: User to optimize preferences for
            
        Returns:
            True if optimization was successful
        """
        try:
            preferences = self.user_store.get_preferences(user_id)
            history = self.history_store.get_history(user_id)
            
            if not preferences or not history:
                return False
            
            # Analyze recent feedback patterns
            recent_exchanges = history.get_recent_exchanges(20)
            if len(recent_exchanges) < 5:
                return False  # Not enough data
            
            # Calculate feedback trends
            positive_feedback = sum(
                1 for ex in recent_exchanges 
                if ex.user_feedback == UserFeedback.THUMB_UP
            )
            negative_feedback = sum(
                1 for ex in recent_exchanges 
                if ex.user_feedback == UserFeedback.THUMB_DOWN
            )
            
            total_feedback = positive_feedback + negative_feedback
            if total_feedback == 0:
                return False  # No feedback to analyze
            
            satisfaction_rate = positive_feedback / total_feedback
            
            # Adjust preferences based on satisfaction
            if satisfaction_rate < 0.3:  # Low satisfaction
                # Try different response style
                current_style = preferences.response_style
                if current_style.value == "concise":
                    preferences.response_style = preferences.response_style.__class__("detailed")
                elif current_style.value == "detailed":
                    preferences.response_style = preferences.response_style.__class__("concise")
                
                preferences.updated_at = datetime.now()
                self.user_store.save_preferences(preferences)
                
                logger.info(f"Optimized preferences for user {user_id} due to low satisfaction")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error optimizing preferences for user {user_id}: {e}")
            return False


class DataExporter:
    """Export context manager data for analysis or backup."""
    
    def __init__(self,
                 session_store: SessionStore,
                 history_store: HistoryStore,
                 user_store: UserStore):
        """Initialize with storage backends."""
        self.session_store = session_store
        self.history_store = history_store
        self.user_store = user_store
    
    def export_user_data(self, 
                        user_id: str,
                        output_dir: str,
                        include_sessions: bool = False) -> bool:
        """
        Export all data for a specific user.
        
        Args:
            user_id: User identifier
            output_dir: Directory to save exported data
            include_sessions: Whether to include active session data
            
        Returns:
            True if export was successful
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                "user_id": user_id,
                "export_timestamp": datetime.now().isoformat(),
                "preferences": None,
                "conversation_history": None,
                "active_sessions": []
            }
            
            # Export preferences
            preferences = self.user_store.get_preferences(user_id)
            if preferences:
                export_data["preferences"] = preferences.to_dict()
            
            # Export conversation history
            history = self.history_store.get_history(user_id)
            if history:
                export_data["conversation_history"] = history.to_dict()
            
            # Export to JSON file
            output_file = output_path / f"user_data_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported user data for {user_id} to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting user data for {user_id}: {e}")
            return False
    
    def export_analytics_report(self, 
                               output_file: str,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> bool:
        """
        Export analytics report as CSV.
        
        Args:
            output_file: Path to save the CSV report
            start_date: Optional start date for report
            end_date: Optional end date for report
            
        Returns:
            True if export was successful
        """
        try:
            # This would require iterating through users, which depends on storage backend
            # For now, create a placeholder report structure
            
            report_data = [
                ["metric", "value", "timestamp"],
                ["total_users", "0", datetime.now().isoformat()],
                ["active_sessions", "0", datetime.now().isoformat()],
                ["avg_satisfaction", "0.0", datetime.now().isoformat()]
            ]
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(report_data)
            
            logger.info(f"Exported analytics report to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting analytics report: {e}")
            return False


class DataMigration:
    """Handle data migration and format conversion."""
    
    def __init__(self):
        """Initialize data migration utilities."""
        pass
    
    def migrate_conversation_format(self,
                                  old_data: Dict[str, Any],
                                  target_version: str = "2.0") -> Dict[str, Any]:
        """
        Migrate conversation data to new format.
        
        Args:
            old_data: Old format conversation data
            target_version: Target format version
            
        Returns:
            Migrated data dictionary
        """
        try:
            if target_version == "2.0":
                # Add any new fields that might be missing
                migrated_data = old_data.copy()
                
                # Ensure all exchanges have required fields
                if "conversation_log" in migrated_data:
                    for exchange in migrated_data["conversation_log"]:
                        if "user_feedback" not in exchange:
                            exchange["user_feedback"] = "none"
                        if "diagram_ids_used" not in exchange:
                            exchange["diagram_ids_used"] = []
                
                migrated_data["format_version"] = target_version
                migrated_data["migrated_at"] = datetime.now().isoformat()
                
                return migrated_data
            
            return old_data
            
        except Exception as e:
            logger.error(f"Error migrating conversation data: {e}")
            return old_data
    
    def validate_data_integrity(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate data integrity and return any issues found.
        
        Args:
            data: Data to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        try:
            # Validate conversation history structure
            if "conversation_log" in data:
                for i, exchange in enumerate(data["conversation_log"]):
                    required_fields = ["query_text", "llm_response", "timestamp"]
                    for field in required_fields:
                        if field not in exchange:
                            errors.append(f"Exchange {i} missing required field: {field}")
                    
                    # Validate timestamp format
                    if "timestamp" in exchange:
                        try:
                            datetime.fromisoformat(exchange["timestamp"])
                        except ValueError:
                            errors.append(f"Exchange {i} has invalid timestamp format")
            
            # Validate preferences structure
            if "response_style" in data:
                valid_styles = ["concise", "detailed", "balanced"]
                if data["response_style"] not in valid_styles:
                    errors.append(f"Invalid response style: {data['response_style']}")
            
            return errors
            
        except Exception as e:
            logger.error(f"Error validating data integrity: {e}")
            return [f"Validation error: {str(e)}"]


# Utility functions for common tasks

def generate_session_id(user_id: str) -> str:
    """Generate a unique session ID."""
    import uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"{user_id}_{timestamp}_{unique_id}"


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Simple word-based similarity (can be enhanced with more sophisticated methods)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def sanitize_user_input(text: str) -> str:
    """
    Sanitize user input for safe storage and processing.
    
    Args:
        text: Raw user input
        
    Returns:
        Sanitized text
    """
    import re
    
    try:
        # Remove potential harmful characters
        sanitized = re.sub(r'[<>"\']', '', text)
        
        # Limit length
        sanitized = truncate_text(sanitized, 1000)
        
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized.strip()
        
    except Exception as e:
        logger.error(f"Error sanitizing user input: {e}")
        return text  # Return original if sanitization fails


def format_timestamp(dt: datetime, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime object to string.
    
    Args:
        dt: Datetime object
        format_string: Format string
        
    Returns:
        Formatted timestamp string
    """
    try:
        return dt.strftime(format_string)
    except Exception as e:
        logger.error(f"Error formatting timestamp: {e}")
        return str(dt)
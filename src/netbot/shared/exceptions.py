"""
Common exceptions for NetBot V2.
"""


class NetBotError(Exception):
    """Base exception for all NetBot errors."""
    pass


class ConfigurationError(NetBotError):
    """Raised when there are configuration issues."""
    pass


class DatabaseError(NetBotError):
    """Raised when database operations fail."""
    pass


class AIServiceError(NetBotError):
    """Raised when AI service calls fail."""
    pass


class ProcessingError(NetBotError):
    """Raised when diagram processing fails."""
    pass


class SearchError(NetBotError):
    """Raised when search operations fail."""
    pass


class ValidationError(NetBotError):
    """Raised when data validation fails."""
    pass


class AuthenticationError(NetBotError):
    """Raised when authentication fails."""
    pass


class RateLimitError(NetBotError):
    """Raised when rate limits are exceeded."""
    pass


class CacheError(NetBotError):
    """Raised when cache operations fail."""
    pass


class StorageError(NetBotError):
    """Raised when storage operations fail."""
    pass


class RepositoryError(NetBotError):
    """Raised when repository operations fail."""
    pass


class AIError(NetBotError):
    """Raised when AI operations fail."""
    pass


class ServiceError(NetBotError):
    """Raised when service operations fail."""
    pass
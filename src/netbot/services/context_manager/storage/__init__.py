"""
Storage backends for Context Manager.

Provides pluggable storage backends for different deployment needs:
- InMemoryStorage: Fast, ephemeral storage for development
- ExternalStorage: Persistent storage with database backing
"""

from .backends import InMemoryStorage, ExternalStorage

__all__ = [
    "InMemoryStorage",
    "ExternalStorage",
]
"""
Query processing for TextRAG search.

Handles query expansion, preprocessing, and optimization
for improved search performance.
"""

import re
from typing import List, Set, Optional
from dataclasses import dataclass

from ....shared import get_logger
from ..models import SearchQuery


@dataclass
class ProcessedQuery:
    """Represents a processed search query."""
    original_text: str
    text: str
    expanded_terms: List[str]
    entities: List[str]
    keywords: List[str]
    intent: Optional[str]


class QueryProcessor:
    """
    Processes and optimizes search queries.
    
    Handles query expansion, entity extraction, and preprocessing
    to improve search accuracy and relevance.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Common stopwords for query processing
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 
            'the', 'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 
            'how', 'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these',
            'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time',
            'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first',
            'been', 'call', 'who', 'oil', 'sit', 'now', 'find', 'down', 'day',
            'did', 'get', 'come', 'made', 'may', 'part'
        }
        
        # Technical terms that shouldn't be filtered
        self.technical_terms = {
            'api', 'cpu', 'gpu', 'ram', 'sql', 'xml', 'json', 'http', 'https',
            'tcp', 'udp', 'ip', 'dns', 'ssl', 'tls', 'vpn', 'lan', 'wan',
            'router', 'switch', 'firewall', 'load balancer', 'database',
            'server', 'client', 'network', 'security', 'authentication',
            'authorization', 'encryption', 'protocol', 'port', 'endpoint'
        }
        
        self.logger.info("Query Processor initialized")
    
    async def process_query(self, query: SearchQuery) -> ProcessedQuery:
        """
        Process and optimize a search query.
        
        Args:
            query: Original search query
            
        Returns:
            Processed query with expansions and metadata
        """
        original_text = query.text
        
        # Basic text cleaning
        cleaned_text = self._clean_query_text(original_text)
        
        # Extract entities
        entities = self._extract_entities(cleaned_text)
        
        # Extract keywords
        keywords = self._extract_keywords(cleaned_text)
        
        # Expand query terms
        expanded_terms = self._expand_query_terms(cleaned_text)
        
        # Detect query intent
        intent = self._detect_intent(cleaned_text)
        
        # Apply query optimization
        optimized_text = self._optimize_query(cleaned_text, keywords, entities)
        
        processed = ProcessedQuery(
            original_text=original_text,
            text=optimized_text,
            expanded_terms=expanded_terms,
            entities=entities,
            keywords=keywords,
            intent=intent
        )
        
        self.logger.debug(f"Processed query: '{original_text}' -> '{optimized_text}'")
        return processed
    
    def _clean_query_text(self, text: str) -> str:
        """Clean and normalize query text."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep technical terms
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from query text."""
        entities = []
        
        # Simple regex-based entity extraction
        # In production, this would use spaCy or similar NLP library
        
        # Network entities
        network_patterns = [
            r'\b(?:firewall|router|switch|server|database|load\s*balancer)\b',
            r'\b(?:vpn|dns|dhcp|nat|vlan)\b',
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP addresses
            r'\b[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)*\b'  # Domain names
        ]
        
        for pattern in network_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        # Remove duplicates and clean
        entities = list(set([e.strip() for e in entities if e.strip()]))
        
        return entities[:10]  # Limit to top 10
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from query text."""
        words = text.split()
        
        # Filter out stopwords but keep technical terms
        keywords = []
        for word in words:
            word = word.strip('.,!?;:()[]{}"\'-')
            if len(word) > 2:
                if word.lower() in self.technical_terms or word.lower() not in self.stopwords:
                    keywords.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword)
        
        return unique_keywords[:15]  # Limit to top 15
    
    def _expand_query_terms(self, text: str) -> List[str]:
        """Expand query terms with synonyms and related terms."""
        expanded = []
        
        # Simple synonym expansion
        synonyms = {
            'firewall': ['security gateway', 'network security', 'packet filter'],
            'router': ['gateway', 'routing device'],
            'switch': ['network switch', 'switching hub'],
            'server': ['host', 'machine', 'node'],
            'database': ['db', 'data store', 'repository'],
            'load balancer': ['lb', 'load balancing', 'traffic distribution'],
            'network': ['infrastructure', 'topology', 'connectivity'],
            'security': ['protection', 'safety', 'defense'],
            'configuration': ['config', 'setup', 'settings'],
            'connection': ['link', 'connectivity', 'communication'],
            'protocol': ['standard', 'specification'],
            'authentication': ['auth', 'login', 'verification'],
            'authorization': ['access control', 'permissions']
        }
        
        words = text.split()
        for word in words:
            if word in synonyms:
                expanded.extend(synonyms[word])
        
        return list(set(expanded))
    
    def _detect_intent(self, text: str) -> Optional[str]:
        """Detect the intent of the query."""
        text_lower = text.lower()
        
        # Question patterns
        if any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'which']):
            if any(word in text_lower for word in ['configure', 'setup', 'install']):
                return 'configuration_help'
            elif any(word in text_lower for word in ['connect', 'connection', 'network']):
                return 'connectivity_help'
            elif any(word in text_lower for word in ['security', 'secure', 'protect']):
                return 'security_help'
            else:
                return 'information_request'
        
        # Action patterns
        if any(word in text_lower for word in ['show', 'list', 'find', 'display', 'get']):
            return 'search_request'
        
        if any(word in text_lower for word in ['configure', 'setup', 'create', 'install']):
            return 'configuration_request'
        
        if any(word in text_lower for word in ['troubleshoot', 'debug', 'fix', 'resolve']):
            return 'troubleshooting_request'
        
        # Default
        return 'general_search'
    
    def _optimize_query(self, text: str, keywords: List[str], entities: List[str]) -> str:
        """Optimize query text for better search performance."""
        # Start with cleaned text
        optimized = text
        
        # Boost important entities by duplication
        important_entities = [e for e in entities if len(e) > 3]
        if important_entities:
            # Add most important entity again for boosting
            main_entity = important_entities[0]
            if main_entity not in optimized:
                optimized = f"{main_entity} {optimized}"
        
        # Add important keywords if missing
        important_keywords = ['firewall', 'router', 'switch', 'server', 'database', 'network', 'security']
        for keyword in important_keywords:
            if keyword in text.lower() and keyword not in optimized:
                optimized = f"{optimized} {keyword}"
        
        return optimized.strip()
    
    def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query suggestions for autocomplete."""
        # This would typically query a database of common queries
        # For now, return some common network-related suggestions
        
        suggestions = [
            "firewall configuration",
            "router setup",
            "network topology",
            "load balancer setup",
            "database connection",
            "security protocols",
            "VPN configuration",
            "network troubleshooting",
            "server deployment",
            "network monitoring"
        ]
        
        partial_lower = partial_query.lower()
        
        # Filter suggestions that match the partial query
        matching = [s for s in suggestions if partial_lower in s.lower()]
        
        return matching[:limit]
"""
Text chunking strategies for TextRAG.

Provides various methods for splitting documents into chunks optimized
for vector storage and semantic search.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ....shared import get_logger
from ..models import ChunkingStrategy, DocumentChunk


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    strategy: ChunkingStrategy
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True


class TextChunker:
    """
    Advanced text chunking with multiple strategies.
    
    Provides intelligent text splitting that preserves semantic coherence
    while optimizing for vector similarity search.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.info("Text Chunker initialized")
    
    def chunk_document(self, 
                      document_id: str,
                      content: str, 
                      config: ChunkConfig) -> List[DocumentChunk]:
        """
        Chunk a document using the specified strategy.
        
        Args:
            document_id: Source document ID
            content: Text content to chunk
            config: Chunking configuration
            
        Returns:
            List of document chunks
        """
        if not content or not content.strip():
            return []
        
        # Choose chunking method based on strategy
        if config.strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(content, config)
        elif config.strategy == ChunkingStrategy.SENTENCE_BOUNDARY:
            chunks = self._chunk_sentence_boundary(content, config)
        elif config.strategy == ChunkingStrategy.PARAGRAPH_BOUNDARY:
            chunks = self._chunk_paragraph_boundary(content, config)
        elif config.strategy == ChunkingStrategy.SEMANTIC_SIMILARITY:
            chunks = self._chunk_semantic_similarity(content, config)
        elif config.strategy == ChunkingStrategy.RECURSIVE_CHARACTER:
            chunks = self._chunk_recursive_character(content, config)
        else:
            # Default to recursive character chunking
            chunks = self._chunk_recursive_character(content, config)
        
        # Convert to DocumentChunk objects
        document_chunks = []
        for i, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            if len(chunk_text.strip()) < config.min_chunk_size:
                continue
                
            chunk = DocumentChunk(
                document_id=document_id,
                content=chunk_text.strip(),
                chunk_index=i,
                start_char=start_pos,
                end_char=end_pos
            )
            
            # Set up chunk relationships
            if i > 0:
                chunk.previous_chunk_id = document_chunks[i-1].chunk_id
                document_chunks[i-1].next_chunk_id = chunk.chunk_id
            
            # Extract additional metadata
            self._enrich_chunk_metadata(chunk)
            
            document_chunks.append(chunk)
        
        self.logger.info(f"Created {len(document_chunks)} chunks for document {document_id}")
        return document_chunks
    
    def _chunk_fixed_size(self, content: str, config: ChunkConfig) -> List[Tuple[str, int, int]]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + config.chunk_size, len(content))
            
            # Try to find a good break point
            if config.preserve_sentences and end < len(content):
                # Look for sentence boundary within the last 10% of chunk
                search_start = max(start + config.chunk_size * 0.9, start)
                sentence_end = self._find_sentence_boundary(content, int(search_start), end)
                if sentence_end > start:
                    end = sentence_end
            
            chunk_text = content[start:end]
            chunks.append((chunk_text, start, end))
            
            # Move start position with overlap
            start = max(start + config.chunk_size - config.chunk_overlap, start + 1)
            
            if start >= len(content):
                break
        
        return chunks
    
    def _chunk_sentence_boundary(self, content: str, config: ChunkConfig) -> List[Tuple[str, int, int]]:
        """Split text at sentence boundaries while respecting size limits."""
        sentences = self._split_sentences(content)
        chunks = []
        current_chunk = ""
        current_start = 0
        sentence_start = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if current_chunk and len(current_chunk + " " + sentence) > config.chunk_size:
                # Save current chunk
                chunks.append((current_chunk, current_start, sentence_start))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, config.chunk_overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_start = max(0, sentence_start - len(overlap_text))
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = sentence_start
            
            sentence_start += len(sentence) + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunks.append((current_chunk, current_start, len(content)))
        
        return chunks
    
    def _chunk_paragraph_boundary(self, content: str, config: ChunkConfig) -> List[Tuple[str, int, int]]:
        """Split text at paragraph boundaries while respecting size limits."""
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        current_start = 0
        para_start = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if current_chunk and len(current_chunk + "\n\n" + paragraph) > config.chunk_size:
                # Save current chunk
                chunks.append((current_chunk, current_start, para_start))
                
                # If single paragraph is too long, split it
                if len(paragraph) > config.chunk_size:
                    para_chunks = self._chunk_sentence_boundary(paragraph, config)
                    for para_chunk, rel_start, rel_end in para_chunks:
                        chunks.append((para_chunk, para_start + rel_start, para_start + rel_end))
                else:
                    # Start new chunk
                    current_chunk = paragraph
                    current_start = para_start
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    current_start = para_start
            
            para_start += len(paragraph) + 2  # +2 for \n\n
        
        # Add final chunk
        if current_chunk:
            chunks.append((current_chunk, current_start, len(content)))
        
        return chunks
    
    def _chunk_recursive_character(self, content: str, config: ChunkConfig) -> List[Tuple[str, int, int]]:
        """
        Recursive character-level chunking with intelligent boundaries.
        
        This is the most sophisticated chunking strategy that tries multiple
        separators in order of preference.
        """
        separators = [
            "\n\n",      # Paragraph breaks
            "\n",        # Line breaks
            ". ",        # Sentence ends
            "! ",        # Exclamation ends
            "? ",        # Question ends
            "; ",        # Semicolons
            ", ",        # Commas
            " ",         # Spaces
            ""           # Character level
        ]
        
        return self._recursive_split(content, separators, config, 0)
    
    def _recursive_split(self, 
                        content: str, 
                        separators: List[str], 
                        config: ChunkConfig,
                        current_start: int = 0) -> List[Tuple[str, int, int]]:
        """Recursively split text using different separators."""
        if len(content) <= config.chunk_size:
            return [(content, current_start, current_start + len(content))]
        
        chunks = []
        
        for separator in separators:
            if separator == "":
                # Character-level splitting as last resort
                return self._chunk_fixed_size(content, config)
            
            splits = content.split(separator)
            if len(splits) == 1:
                continue  # Separator not found, try next one
            
            current_chunk = ""
            chunk_start = current_start
            position = current_start
            
            for i, split in enumerate(splits):
                test_chunk = current_chunk + separator + split if current_chunk else split
                
                if len(test_chunk) <= config.chunk_size:
                    current_chunk = test_chunk
                else:
                    # Current chunk is ready
                    if current_chunk:
                        chunks.append((current_chunk, chunk_start, position))
                    
                    # Handle overlap
                    if config.chunk_overlap > 0 and current_chunk:
                        overlap = self._get_overlap_text(current_chunk, config.chunk_overlap)
                        current_chunk = overlap + separator + split if overlap else split
                        chunk_start = max(chunk_start, position - len(overlap))
                    else:
                        current_chunk = split
                        chunk_start = position
                
                position += len(split)
                if i < len(splits) - 1:
                    position += len(separator)
            
            # Add final chunk
            if current_chunk:
                chunks.append((current_chunk, chunk_start, current_start + len(content)))
            
            return chunks
        
        # Fallback to fixed size if no separators work
        return self._chunk_fixed_size(content, config)
    
    def _chunk_semantic_similarity(self, content: str, config: ChunkConfig) -> List[Tuple[str, int, int]]:
        """
        Semantic similarity-based chunking.
        
        This is a placeholder for advanced semantic chunking that would
        use embeddings to determine optimal split points.
        """
        # For now, fall back to sentence boundary chunking
        # In a full implementation, this would:
        # 1. Split into sentences
        # 2. Generate embeddings for each sentence
        # 3. Use similarity metrics to group related sentences
        # 4. Create chunks based on semantic coherence
        
        self.logger.warning("Semantic similarity chunking not fully implemented, using sentence boundary")
        return self._chunk_sentence_boundary(content, config)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Simple sentence splitting - could be improved with NLTK
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within a range."""
        search_text = text[start:end]
        
        # Look for sentence endings
        for pattern in [r'\.(?=\s+[A-Z])', r'[!?](?=\s)', r'\.(?=\s*$)']:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Return the position after the last match
                last_match = matches[-1]
                return start + last_match.end()
        
        return end
    
    def _get_overlap_text(self, chunk: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        if len(chunk) <= overlap_size:
            return chunk
        
        # Try to find a good breaking point for overlap
        overlap_text = chunk[-overlap_size:]
        
        # Find the start of a sentence or word
        space_pos = overlap_text.find(' ')
        if space_pos != -1:
            overlap_text = overlap_text[space_pos + 1:]
        
        return overlap_text
    
    def _enrich_chunk_metadata(self, chunk: DocumentChunk) -> None:
        """Add metadata to a chunk for better search optimization."""
        content = chunk.content
        
        # Extract keywords (simple approach - could use NLP libraries)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        chunk.keywords = [word for word, freq in top_keywords]
        
        # Extract potential named entities (basic approach)
        # This is a simple regex-based approach - could use spaCy or NLTK
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        chunk.named_entities = list(set(named_entities))[:10]  # Limit to 10
        
        # Calculate basic coherence score based on sentence structure
        sentences = self._split_sentences(content)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            # Normalize to 0-1 range (assume 15 words is optimal)
            chunk.coherence_score = min(1.0, avg_sentence_length / 15.0)
        
        # Set relevance score (would be updated during search)
        chunk.relevance_score = 0.8  # Default high relevance
        
        # Add section title if we can detect it
        first_line = content.split('\n')[0].strip()
        if len(first_line) < 100 and any(char.isupper() for char in first_line):
            chunk.section_title = first_line
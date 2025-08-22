"""
Hybrid chunking strategy for diagram + context approach.

Creates semantic chunks that combine diagram references with surrounding
contextual text, enabling the hybrid RAG workflow.
"""

import re
import uuid
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .models import DiagramChunk, ChunkType


@dataclass
class DocumentSection:
    """A section of text that may contain diagram references."""
    text: str
    start_pos: int
    end_pos: int
    diagram_refs: List[str] = None  # Diagram IDs found in this section
    
    def __post_init__(self):
        if self.diagram_refs is None:
            self.diagram_refs = []


class HybridChunker:
    """
    Creates chunks optimized for hybrid RAG retrieval.
    
    The strategy is to identify diagram references in text and create
    chunks that combine the diagram ID with surrounding contextual text.
    This enables vector search to find relevant chunks, which then
    trigger graph-based retrieval for the associated diagrams.
    """
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 128,
                 context_window: int = 256):
        """
        Initialize the hybrid chunker.
        
        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between adjacent chunks
            context_window: Size of context around diagram references
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_window = context_window
        
        # Patterns to identify diagram references in text
        self.diagram_patterns = [
            r'diagram[_\s]+(\w+)',           # "diagram_001", "diagram 1"
            r'figure[_\s]+(\w+)',            # "figure_1", "figure 2"
            r'chart[_\s]+(\w+)',             # "chart_a", "chart 1"
            r'graph[_\s]+(\w+)',             # "graph_001"
            r'image[_\s]+(\w+)',             # "image_1"
            r'(\w+\.(?:png|jpg|jpeg|svg))',  # "network_diagram.png"
        ]
    
    def chunk_document(self, 
                      text: str, 
                      source_document: str = None,
                      known_diagram_ids: List[str] = None) -> List[DiagramChunk]:
        """
        Chunk a document using hybrid strategy.
        
        Args:
            text: The document text to chunk
            source_document: Name/path of source document
            known_diagram_ids: List of known diagram IDs to look for
            
        Returns:
            List of DiagramChunk objects
        """
        if known_diagram_ids is None:
            known_diagram_ids = []
        
        # Step 1: Find all diagram references in the text
        diagram_positions = self._find_diagram_references(text, known_diagram_ids)
        
        # Step 2: Create sections around diagram references
        diagram_sections = self._create_diagram_sections(text, diagram_positions)
        
        # Step 3: Create chunks for diagram sections
        chunks = []
        for section in diagram_sections:
            section_chunks = self._chunk_diagram_section(section, source_document)
            chunks.extend(section_chunks)
        
        # Step 4: Handle remaining text without diagram references
        remaining_text = self._get_text_without_diagrams(text, diagram_sections)
        if remaining_text.strip():
            text_chunks = self._chunk_plain_text(remaining_text, source_document)
            chunks.extend(text_chunks)
        
        return chunks
    
    def _find_diagram_references(self, 
                               text: str, 
                               known_diagram_ids: List[str]) -> List[Tuple[int, int, str]]:
        """
        Find all diagram references in text.
        
        Returns:
            List of (start_pos, end_pos, diagram_id) tuples
        """
        references = []
        
        # Look for known diagram IDs first (exact matches)
        for diagram_id in known_diagram_ids:
            pattern = rf'\b{re.escape(diagram_id)}\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                references.append((match.start(), match.end(), diagram_id))
        
        # Look for diagram patterns
        for pattern in self.diagram_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                if match.groups():
                    # Extract diagram ID from capture group
                    diagram_id = match.group(1)
                else:
                    # Use full match as diagram ID
                    diagram_id = match.group(0)
                
                # Clean up diagram ID
                diagram_id = re.sub(r'[^\w.-]', '_', diagram_id)
                references.append((start, end, diagram_id))
        
        # Remove duplicates and sort by position
        references = list(set(references))
        references.sort(key=lambda x: x[0])
        
        return references
    
    def _create_diagram_sections(self, 
                               text: str, 
                               diagram_positions: List[Tuple[int, int, str]]) -> List[DocumentSection]:
        """
        Create sections around diagram references with context.
        """
        sections = []
        
        for start_pos, end_pos, diagram_id in diagram_positions:
            # Calculate context window around diagram reference
            context_start = max(0, start_pos - self.context_window)
            context_end = min(len(text), end_pos + self.context_window)
            
            # Try to break at sentence boundaries
            context_start = self._find_sentence_boundary(text, context_start, direction='backward')
            context_end = self._find_sentence_boundary(text, context_end, direction='forward')
            
            section_text = text[context_start:context_end]
            section = DocumentSection(
                text=section_text,
                start_pos=context_start,
                end_pos=context_end,
                diagram_refs=[diagram_id]
            )
            sections.append(section)
        
        # Merge overlapping sections
        sections = self._merge_overlapping_sections(sections)
        
        return sections
    
    def _find_sentence_boundary(self, text: str, pos: int, direction: str = 'forward') -> int:
        """Find the nearest sentence boundary."""
        sentence_endings = '.!?'
        
        if direction == 'forward':
            for i in range(pos, min(len(text), pos + 100)):
                if text[i] in sentence_endings and i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
            return pos
        else:  # backward
            for i in range(pos, max(0, pos - 100), -1):
                if text[i] in sentence_endings and i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
            return pos
    
    def _merge_overlapping_sections(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Merge sections that overlap significantly."""
        if not sections:
            return sections
        
        merged = []
        current = sections[0]
        
        for next_section in sections[1:]:
            # Check for overlap
            overlap_start = max(current.start_pos, next_section.start_pos)
            overlap_end = min(current.end_pos, next_section.end_pos)
            overlap_size = max(0, overlap_end - overlap_start)
            
            # If overlap is significant, merge sections
            if overlap_size > self.chunk_overlap:
                # Merge sections
                merged_start = min(current.start_pos, next_section.start_pos)
                merged_end = max(current.end_pos, next_section.end_pos)
                merged_text = current.text  # Will be recalculated if needed
                merged_refs = list(set(current.diagram_refs + next_section.diagram_refs))
                
                current = DocumentSection(
                    text=merged_text,
                    start_pos=merged_start,
                    end_pos=merged_end,
                    diagram_refs=merged_refs
                )
            else:
                merged.append(current)
                current = next_section
        
        merged.append(current)
        return merged
    
    def _chunk_diagram_section(self, 
                             section: DocumentSection, 
                             source_document: str) -> List[DiagramChunk]:
        """Create chunks for a section containing diagram references."""
        chunks = []
        
        # For sections with diagram references, create context-aware chunks
        text = section.text
        
        # If section is small enough, create single chunk
        if len(text) <= self.chunk_size:
            chunk = self._create_diagram_chunk(
                text=text,
                diagram_ids=section.diagram_refs,
                source_document=source_document
            )
            chunks.append(chunk)
        else:
            # Split into smaller chunks while preserving diagram context
            chunk_texts = self._split_text_with_overlap(text)
            for chunk_text in chunk_texts:
                chunk = self._create_diagram_chunk(
                    text=chunk_text,
                    diagram_ids=section.diagram_refs,
                    source_document=source_document
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_diagram_chunk(self, 
                            text: str, 
                            diagram_ids: List[str], 
                            source_document: str) -> DiagramChunk:
        """Create a chunk that references diagrams."""
        # Use first diagram ID as primary reference
        primary_diagram_id = diagram_ids[0] if diagram_ids else None
        
        # Extract text before and after diagram mention
        text_before, text_after = self._split_text_around_diagram(text, primary_diagram_id)
        
        chunk = DiagramChunk(
            chunk_id=str(uuid.uuid4()),
            diagram_id=primary_diagram_id,
            text_content=text,
            chunk_type=ChunkType.DIAGRAM_WITH_CONTEXT,
            source_document=source_document,
            text_before_diagram=text_before,
            text_after_diagram=text_after,
            properties={
                "all_diagram_ids": diagram_ids,
                "diagram_count": len(diagram_ids)
            }
        )
        
        return chunk
    
    def _split_text_around_diagram(self, text: str, diagram_id: str) -> Tuple[str, str]:
        """Split text into before and after diagram mention."""
        if not diagram_id:
            return text, ""
        
        # Find diagram mention
        pattern = rf'\b{re.escape(diagram_id)}\b'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            split_pos = match.start()
            before = text[:split_pos].strip()
            after = text[match.end():].strip()
            return before, after
        else:
            return text, ""
    
    def _chunk_plain_text(self, 
                         text: str, 
                         source_document: str) -> List[DiagramChunk]:
        """Create chunks for text without diagram references."""
        chunks = []
        chunk_texts = self._split_text_with_overlap(text)
        
        for chunk_text in chunk_texts:
            chunk = DiagramChunk(
                chunk_id=str(uuid.uuid4()),
                text_content=chunk_text,
                chunk_type=ChunkType.PURE_TEXT,
                source_document=source_document
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_text_with_overlap(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Find last space before end position
                while end > start and not text[end].isspace():
                    end -= 1
                if end == start:  # No space found, break at chunk_size
                    end = start + self.chunk_size
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                break
        
        return chunks
    
    def _get_text_without_diagrams(self, 
                                  text: str, 
                                  diagram_sections: List[DocumentSection]) -> str:
        """Get text that doesn't overlap with diagram sections."""
        if not diagram_sections:
            return text
        
        # Sort sections by position
        sections = sorted(diagram_sections, key=lambda s: s.start_pos)
        
        remaining_parts = []
        last_end = 0
        
        for section in sections:
            # Add text before this section
            if section.start_pos > last_end:
                remaining_parts.append(text[last_end:section.start_pos])
            last_end = max(last_end, section.end_pos)
        
        # Add text after last section
        if last_end < len(text):
            remaining_parts.append(text[last_end:])
        
        return " ".join(part.strip() for part in remaining_parts if part.strip())
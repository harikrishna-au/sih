"""
DOCX content chunking utilities.

Provides intelligent chunking of DOCX content with structure awareness,
paragraph-level source tracking, and configurable chunking strategies.
"""

import re
import logging
from typing import List, Dict, Any
from pathlib import Path

from ..base import ProcessingError
from ...models import ContentChunk, SourceLocation, ContentType
from ...config import ProcessingConfig

logger = logging.getLogger(__name__)


class DOCXChunker:
    """Handles intelligent chunking of DOCX content."""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize DOCX chunker.
        
        Args:
            config: Processing configuration
        """
        self.config = config
    
    def chunk_content(self, content: str, document_id: str, file_path: str) -> List[ContentChunk]:
        """
        Split DOCX content into chunks with paragraph-level source tracking.
        
        Args:
            content: Extracted DOCX content with structure markers
            document_id: Unique identifier for the document
            file_path: Path to the source file
            
        Returns:
            List of ContentChunk objects with source location tracking
            
        Raises:
            ProcessingError: If chunking fails
        """
        try:
            chunks = []
            
            if not content or not content.strip():
                logger.warning(f"No content to chunk for {file_path}")
                return chunks
            
            # Split content by structure markers
            sections = self._split_by_structure(content)
            
            if not sections:
                # Fallback to simple text chunking
                logger.info(f"No structure found in {file_path}, using text-based chunking")
                text_chunks = self._create_text_chunks(
                    content, 
                    self.config.chunk_size, 
                    self.config.chunk_overlap
                )
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk = ContentChunk(
                        chunk_id=f"{document_id}_chunk_{i}",
                        content=chunk_text,
                        content_type=ContentType.TEXT,
                        source_location=SourceLocation(
                            file_path=file_path,
                            page_number=1,
                            paragraph_index=i,
                            chunk_index=i
                        ),
                        metadata={
                            'chunk_method': 'text_based',
                            'original_length': len(content),
                            'chunk_size': len(chunk_text)
                        }
                    )
                    chunks.append(chunk)
                
                return chunks
            
            # Structure-aware chunking
            current_chunk_parts = []
            current_chunk_size = 0
            chunk_index = 0
            
            for section in sections:
                section_text = section['content']
                section_size = len(section_text)
                
                # If section is too large, split it
                if section_size > self.config.chunk_size:
                    # Save current chunk if it has content
                    if current_chunk_parts:
                        chunk = self._create_chunk_from_parts(
                            current_chunk_parts, document_id, file_path, chunk_index
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        current_chunk_parts = []
                        current_chunk_size = 0
                    
                    # Split large section
                    section_chunks = self._split_large_section(section, document_id, file_path, chunk_index)
                    chunks.extend(section_chunks)
                    chunk_index += len(section_chunks)
                
                # If adding this section would exceed chunk size, finalize current chunk
                elif current_chunk_size + section_size > self.config.chunk_size and current_chunk_parts:
                    chunk = self._create_chunk_from_parts(
                        current_chunk_parts, document_id, file_path, chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_parts = [section]
                    current_chunk_size = section_size
                
                else:
                    # Add section to current chunk
                    current_chunk_parts.append(section)
                    current_chunk_size += section_size
            
            # Handle remaining content
            if current_chunk_parts:
                chunk = self._create_chunk_from_parts(
                    current_chunk_parts, document_id, file_path, chunk_index
                )
                chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking failed for {file_path}: {e}")
            raise ProcessingError(
                f"Failed to chunk DOCX content from {file_path}: {str(e)}",
                file_path=file_path,
                cause=e
            )
    
    def _split_by_structure(self, content: str) -> List[Dict[str, Any]]:
        """
        Split content by structure markers and return structured sections.
        
        Args:
            content: Content with structure markers
            
        Returns:
            List of section dictionaries with content and metadata
        """
        sections = []
        
        # Split by structure markers
        parts = re.split(r'\[(HEADING|TITLE|SUBTITLE|PARAGRAPH|TABLE|CAPTION|QUOTE) [^\]]*\]', content)
        
        if len(parts) <= 1:
            return []  # No structure markers found
        
        # Process parts (alternating between markers and content)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                marker = parts[i]
                section_content = parts[i + 1].strip()
                
                if section_content:
                    # Parse marker
                    marker_match = re.match(r'(HEADING|TITLE|SUBTITLE|PARAGRAPH|TABLE|CAPTION|QUOTE)\s*(\d+)?', marker)
                    if marker_match:
                        section_type = marker_match.group(1)
                        section_number = marker_match.group(2)
                        
                        sections.append({
                            'type': section_type,
                            'number': int(section_number) if section_number else None,
                            'content': section_content,
                            'marker': marker
                        })
        
        return sections
    
    def _create_chunk_from_parts(
        self, 
        parts: List[Dict[str, Any]], 
        document_id: str, 
        file_path: str, 
        chunk_index: int
    ) -> ContentChunk:
        """
        Create a content chunk from multiple parts.
        
        Args:
            parts: List of section parts
            document_id: Document identifier
            file_path: Source file path
            chunk_index: Index of the chunk
            
        Returns:
            ContentChunk object
        """
        # Combine content from all parts
        chunk_content = "\n\n".join(part['content'] for part in parts)
        
        # Determine primary section type and number
        primary_section = parts[0] if parts else None
        section_types = [part['type'] for part in parts]
        
        # Create metadata
        metadata = {
            'chunk_method': 'structure_aware',
            'section_count': len(parts),
            'section_types': section_types,
            'chunk_size': len(chunk_content)
        }
        
        if primary_section:
            metadata['primary_section_type'] = primary_section['type']
            if primary_section['number'] is not None:
                metadata['primary_section_number'] = primary_section['number']
        
        # Create source location
        source_location = SourceLocation(
            file_path=file_path,
            page_number=1,  # DOCX doesn't have clear page boundaries
            paragraph_index=primary_section['number'] if primary_section and primary_section['number'] else chunk_index,
            chunk_index=chunk_index
        )
        
        return ContentChunk(
            chunk_id=f"{document_id}_chunk_{chunk_index}",
            content=chunk_content,
            content_type=ContentType.TEXT,
            source_location=source_location,
            metadata=metadata
        )
    
    def _split_large_section(
        self, 
        section: Dict[str, Any], 
        document_id: str, 
        file_path: str, 
        start_chunk_index: int
    ) -> List[ContentChunk]:
        """
        Split a large section into multiple chunks.
        
        Args:
            section: Section dictionary
            document_id: Document identifier
            file_path: Source file path
            start_chunk_index: Starting chunk index
            
        Returns:
            List of ContentChunk objects
        """
        chunks = []
        section_content = section['content']
        
        # Split into text chunks
        text_chunks = self._create_text_chunks(
            section_content,
            self.config.chunk_size,
            self.config.chunk_overlap
        )
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_index = start_chunk_index + i
            
            metadata = {
                'chunk_method': 'large_section_split',
                'original_section_type': section['type'],
                'section_part': f"{i + 1}/{len(text_chunks)}",
                'chunk_size': len(chunk_text)
            }
            
            if section['number'] is not None:
                metadata['original_section_number'] = section['number']
            
            source_location = SourceLocation(
                file_path=file_path,
                page_number=1,
                paragraph_index=section['number'] if section['number'] else chunk_index,
                chunk_index=chunk_index
            )
            
            chunk = ContentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                content=chunk_text,
                content_type=ContentType.TEXT,
                source_location=source_location,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_text_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Create overlapping text chunks from content.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Overlap size in characters
            
        Returns:
            List of text chunks
        """
        if not text or chunk_size <= 0:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at word boundary
            if end < len(text):
                # Look for word boundary within the last 10% of chunk
                boundary_start = end - max(1, chunk_size // 10)
                word_boundary = text.rfind(' ', boundary_start, end)
                
                if word_boundary > start:
                    end = word_boundary
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def get_chunking_stats(self, chunks: List[ContentChunk]) -> Dict[str, Any]:
        """
        Get statistics about the chunking process.
        
        Args:
            chunks: List of created chunks
            
        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {'total_chunks': 0}
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        section_types = []
        
        for chunk in chunks:
            if 'section_types' in chunk.metadata:
                section_types.extend(chunk.metadata['section_types'])
            elif 'primary_section_type' in chunk.metadata:
                section_types.append(chunk.metadata['primary_section_type'])
        
        return {
            'total_chunks': len(chunks),
            'average_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'section_types_found': list(set(section_types)),
            'total_content_length': sum(chunk_sizes)
        }
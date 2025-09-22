"""
Text file processor for plain text documents.

Handles .txt files by chunking them into smaller segments
for embedding and retrieval.
"""

import logging
from typing import List, Optional
from pathlib import Path
import uuid

from .base import DocumentProcessor, ProcessingError
from ..models import (
    DocumentContent, ContentChunk, ContentType, SourceLocation, 
    ProcessingResult, ValidationResult, DocumentMetadata
)
from ..config import ProcessingConfig

logger = logging.getLogger(__name__)


class TextProcessor(DocumentProcessor):
    """
    Processor for plain text files (.txt).
    
    Chunks text files into smaller segments based on configuration
    and creates ContentChunk objects for embedding generation.
    """
    
    def __init__(self, config: ProcessingConfig):
        """Initialize text processor with configuration."""
        super().__init__(config)
        self.supported_formats = ['txt']
        self.chunk_size = getattr(config, 'chunk_size', 1000)
        self.chunk_overlap = getattr(config, 'chunk_overlap', 200)
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate if the file can be processed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            ValidationResult with validation status
        """
        try:
            path = Path(file_path)
            
            # Check if file exists and is readable
            if not path.exists():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File does not exist: {file_path}"
                )
            
            if not path.is_file():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Path is not a file: {file_path}"
                )
            
            # Check file extension
            if path.suffix.lower() != '.txt':
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Not a text file: {file_path}"
                )
            
            # Check file size
            if self._is_file_too_large(file_path):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File too large: {file_path}"
                )
            
            # Try to read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few characters to check if it's readable
                f.read(100)
            
            return ValidationResult(
                is_valid=True,
                file_format='txt',
                file_size=self._get_file_size(file_path)
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Text file validation failed: {str(e)}"
            )
    
    def extract_content(self, file_path: str) -> str:
        """
        Extract raw content from the text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Raw text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ProcessingError(f"Failed to extract content from {file_path}: {str(e)}", file_path)
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract metadata from the text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            DocumentMetadata with file information
        """
        path = Path(file_path)
        
        return DocumentMetadata(
            title=path.stem,
            author=None,
            creation_date=None,
            file_size=self._get_file_size(file_path),
            page_count=None,
            language=None,
            format_version=None
        )
    
    def chunk_content(self, content: str, document_id: str, file_path: str) -> List[ContentChunk]:
        """
        Split content into chunks for embedding and indexing.
        
        Args:
            content: Text content to chunk
            document_id: Unique identifier for the document
            file_path: Original file path for source location
            
        Returns:
            List of ContentChunk objects
        """
        return self._create_text_chunks(content, document_id, file_path)
    
    def get_content_type(self) -> ContentType:
        """
        Get the content type handled by this processor.
        
        Returns:
            ContentType.TEXT
        """
        return ContentType.TEXT
    
    def _create_text_chunks(self, text: str, document_id: str, file_path: str) -> List[ContentChunk]:
        """
        Create chunks from text content.
        
        Args:
            text: Full text content
            document_id: Document identifier
            file_path: Path to the source file
            
        Returns:
            List of ContentChunk objects
        """
        chunks = []
        
        # Simple chunking by character count with overlap
        text_length = len(text)
        start = 0
        chunk_index = 0
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at word boundaries if possible
            if end < text_length:
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only create chunk if it has content
                chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
                
                # Create source location (for text files, we don't have page numbers)
                source_location = SourceLocation(
                    file_path=file_path,
                    page_number=None
                )
                
                # Create content chunk (without embedding - will be generated later)
                chunk = ContentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=chunk_text,
                    content_type=ContentType.TEXT,
                    embedding=None,  # Will be generated by embedding pipeline
                    source_location=source_location,
                    metadata={
                        "chunk_index": chunk_index,
                        "start_position": start,
                        "end_position": end,
                        "chunk_length": len(chunk_text)
                    },
                    confidence_score=1.0  # Text extraction is always confident
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next chunk with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            # Prevent infinite loop
            if start >= text_length:
                break
        
        return chunks
    
    def _get_current_time(self) -> float:
        """Get current time for performance measurement."""
        import time
        return time.time()
    

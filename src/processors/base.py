"""
Base interfaces and abstract classes for document processors.

Defines the common interface that all document processors must implement
to ensure consistent behavior across different file formats.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models import (
    DocumentContent, ContentChunk, ValidationResult, 
    ProcessingResult, ContentType, DocumentMetadata
)
from ..config import ProcessingConfig


class DocumentProcessor(ABC):
    """
    Abstract base class for all document processors.
    
    Each file format (PDF, DOCX, images, audio) should implement
    this interface to provide consistent processing capabilities.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.supported_formats: List[str] = []
    
    @abstractmethod
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate that the file can be processed by this processor.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        pass
    
    @abstractmethod
    def extract_content(self, file_path: str) -> str:
        """
        Extract raw content from the file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Raw extracted content as string
            
        Raises:
            ProcessingError: If content extraction fails
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract metadata from the file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            DocumentMetadata with extracted information
        """
        pass
    
    @abstractmethod
    def chunk_content(self, content: str, document_id: str, file_path: str) -> List[ContentChunk]:
        """
        Split content into chunks for embedding and indexing.
        
        Args:
            content: Processed content to chunk
            document_id: Unique identifier for the document
            file_path: Original file path for source location
            
        Returns:
            List of ContentChunk objects
        """
        pass
    
    def process_document(self, file_path: str, document_id: Optional[str] = None) -> ProcessingResult:
        """
        Complete document processing pipeline.
        
        Args:
            file_path: Path to the file to process
            document_id: Optional document ID, generated if not provided
            
        Returns:
            ProcessingResult with processed document or error information
        """
        import time
        import uuid
        
        start_time = time.time()
        
        try:
            # Validate file
            validation = self.validate_file(file_path)
            if not validation.is_valid:
                return ProcessingResult(
                    success=False,
                    error_message=validation.error_message,
                    processing_time=time.time() - start_time
                )
            
            # Generate document ID if not provided
            if document_id is None:
                import hashlib
                import time
                # Create a simple, non-UUID document ID
                timestamp = str(int(time.time()))
                file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
                document_id = f"doc_{timestamp}_{file_hash}"
            
            # Extract content and metadata
            raw_content = self.extract_content(file_path)
            metadata = self.extract_metadata(file_path)
            
            # Process content (subclasses can override this)
            processed_content = self.process_content(raw_content)
            
            # Create chunks
            chunks = self.chunk_content(processed_content, document_id, file_path)
            
            # Create document content object
            document_content = DocumentContent(
                document_id=document_id,
                content_type=self.get_content_type(),
                raw_content=raw_content,
                processed_content=processed_content,
                metadata=metadata,
                chunks=chunks,
                file_path=file_path
            )
            
            return ProcessingResult(
                success=True,
                document_content=document_content,
                processing_time=time.time() - start_time,
                chunks_created=len(chunks)
            )
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def process_content(self, raw_content: str) -> str:
        """
        Process raw content (can be overridden by subclasses).
        
        Args:
            raw_content: Raw extracted content
            
        Returns:
            Processed content ready for chunking
        """
        # Default implementation: basic text cleaning
        return raw_content.strip()
    
    @abstractmethod
    def get_content_type(self) -> ContentType:
        """
        Get the content type handled by this processor.
        
        Returns:
            ContentType enum value
        """
        pass
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of file formats supported by this processor.
        
        Returns:
            List of supported file extensions (without dots)
        """
        return self.supported_formats.copy()
    
    def _create_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """
        Create a unique chunk ID.
        
        Args:
            document_id: Parent document ID
            chunk_index: Index of the chunk within the document
            
        Returns:
            Unique chunk identifier
        """
        return f"{document_id}_chunk_{chunk_index:04d}"
    
    def _get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes
        """
        return Path(file_path).stat().st_size
    
    def _is_file_too_large(self, file_path: str) -> bool:
        """
        Check if file exceeds maximum size limit.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is too large
        """
        max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
        return self._get_file_size(file_path) > max_size_bytes


class ProcessingError(Exception):
    """Exception raised during document processing."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, cause: Optional[Exception] = None):
        self.message = message
        self.file_path = file_path
        self.cause = cause
        super().__init__(self.message)


class UnsupportedFormatError(ProcessingError):
    """Exception raised when file format is not supported."""
    pass


class FileValidationError(ProcessingError):
    """Exception raised when file validation fails."""
    pass


class ContentExtractionError(ProcessingError):
    """Exception raised when content extraction fails."""
    pass
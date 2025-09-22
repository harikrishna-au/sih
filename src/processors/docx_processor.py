"""
DOCX document processor for extracting structured content, metadata, and creating content chunks.

Uses python-docx for DOCX parsing with support for text extraction, headings,
tables, document structure, and formatting metadata preservation.
"""

from pathlib import Path
from typing import List

from .base import DocumentProcessor
from .docx import DOCXValidator, DOCXExtractor, DOCXChunker
from ..models import DocumentMetadata, ContentChunk, ValidationResult, ContentType
from ..config import ProcessingConfig


class DOCXProcessor(DocumentProcessor):
    """
    DOCX document processor that extracts structured content and metadata.
    
    Features:
    - Text extraction with structure preservation
    - Heading hierarchy detection and preservation
    - Table content extraction with structure
    - Document metadata extraction
    - Paragraph-level source location tracking
    - Configurable content chunking with structure awareness
    - Formatting metadata preservation
    """
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.supported_formats = ['docx']
        
        # Initialize components
        self.validator = DOCXValidator()
        self.extractor = DOCXExtractor()
        self.chunker = DOCXChunker(config)
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate that the file is a readable DOCX document.
        
        Args:
            file_path: Path to the DOCX file to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        return self.validator.validate_file(file_path)
    
    def extract_content(self, file_path: str) -> str:
        """
        Extract structured content from the DOCX document.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted content with structure markers
        """
        return self.extractor.extract_content(file_path)
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract metadata from the DOCX document.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            DocumentMetadata object with extracted metadata
        """
        metadata = self.extractor.extract_metadata(file_path)
        
        # Add file size
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            metadata.file_size = file_path_obj.stat().st_size
        
        return metadata
    
    def chunk_content(self, content: str, document_id: str, file_path: str) -> List[ContentChunk]:
        """
        Split DOCX content into chunks with paragraph-level source tracking.
        
        Args:
            content: Extracted DOCX content with structure markers
            document_id: Unique identifier for the document
            file_path: Path to the source file
            
        Returns:
            List of ContentChunk objects with source location tracking
        """
        return self.chunker.chunk_content(content, document_id, file_path)
    
    def get_content_type(self) -> ContentType:
        """Get the content type handled by this processor."""
        return ContentType.TEXT
    
    def process_content(self, raw_content: str) -> str:
        """
        Process raw DOCX content by cleaning and normalizing text.
        
        Args:
            raw_content: Raw extracted content
            
        Returns:
            Processed and cleaned content
        """
        if not raw_content:
            return ""
        
        # Basic content processing (cleaning is done in extractor)
        content = raw_content.strip()
        
        # Remove excessive blank lines
        import re
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    def get_processing_stats(self) -> dict:
        """
        Get processing statistics and information.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            'processor_type': 'DOCX',
            'supported_formats': self.supported_formats,
            'components': {
                'validator': type(self.validator).__name__,
                'extractor': type(self.extractor).__name__,
                'chunker': type(self.chunker).__name__
            },
            'config': {
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap
            }
        }
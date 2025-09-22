"""
PDF document processor for extracting text, metadata, and creating content chunks.

Uses PyPDF2 for PDF parsing and text extraction with support for metadata
extraction and configurable content chunking.
"""

import re
from pathlib import Path
from typing import List, Optional
from datetime import datetime

try:
    import PyPDF2
except ImportError:
    raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2>=3.0.0")

from .base import DocumentProcessor, ProcessingError, ContentExtractionError, FileValidationError
from ..models import (
    DocumentMetadata, ContentChunk, ValidationResult, ContentType, SourceLocation
)
from ..config import ProcessingConfig


class PDFProcessor(DocumentProcessor):
    """
    PDF document processor that extracts text content and metadata.
    
    Features:
    - Text extraction from all pages
    - Metadata extraction (title, author, creation date, page count)
    - Configurable content chunking with overlap
    - Page-level source location tracking
    - File validation and error handling
    """
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.supported_formats = ['pdf']
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate that the file is a readable PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ValidationResult with validation status and any errors
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File not found: {file_path}"
                )
            
            # Check file extension
            if path.suffix.lower() != '.pdf':
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Invalid file extension. Expected .pdf, got {path.suffix}",
                    file_format=path.suffix.lower()
                )
            
            # Check file size
            if self._is_file_too_large(file_path):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File too large. Maximum size: {self.config.max_file_size_mb}MB",
                    file_size=self._get_file_size(file_path)
                )
            
            # Try to open and read the PDF
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    
                    # Check if PDF is encrypted
                    if reader.is_encrypted:
                        return ValidationResult(
                            is_valid=False,
                            error_message="PDF is encrypted and cannot be processed",
                            file_format="pdf"
                        )
                    
                    # Check if PDF has pages
                    if len(reader.pages) == 0:
                        return ValidationResult(
                            is_valid=False,
                            error_message="PDF contains no pages",
                            file_format="pdf"
                        )
                    
                    # Try to extract text from first page to verify readability
                    try:
                        first_page = reader.pages[0]
                        first_page.extract_text()
                    except Exception as e:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"Cannot extract text from PDF: {str(e)}",
                            file_format="pdf"
                        )
            
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Cannot read PDF file: {str(e)}",
                    file_format="pdf"
                )
            
            return ValidationResult(
                is_valid=True,
                file_format="pdf",
                file_size=self._get_file_size(file_path)
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    def extract_content(self, file_path: str) -> str:
        """
        Extract text content from all pages of the PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content from all pages
            
        Raises:
            ContentExtractionError: If text extraction fails
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                if reader.is_encrypted:
                    raise ContentExtractionError(
                        "Cannot extract content from encrypted PDF",
                        file_path=file_path
                    )
                
                extracted_text = []
                
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            # Add page marker for chunking reference
                            extracted_text.append(f"[PAGE {page_num}]\n{page_text}")
                    except Exception as e:
                        # Log warning but continue with other pages
                        print(f"Warning: Could not extract text from page {page_num}: {str(e)}")
                        continue
                
                if not extracted_text:
                    raise ContentExtractionError(
                        "No text content could be extracted from PDF",
                        file_path=file_path
                    )
                
                return "\n\n".join(extracted_text)
                
        except FileNotFoundError:
            raise ContentExtractionError(
                f"PDF file not found: {file_path}",
                file_path=file_path
            )
        except Exception as e:
            raise ContentExtractionError(
                f"Failed to extract content from PDF: {str(e)}",
                file_path=file_path,
                cause=e
            )
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract metadata from the PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            DocumentMetadata with extracted information
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Get basic file information
                file_size = self._get_file_size(file_path)
                page_count = len(reader.pages)
                
                # Extract PDF metadata
                metadata_dict = reader.metadata or {}
                
                # Extract title
                title = None
                if '/Title' in metadata_dict:
                    title = str(metadata_dict['/Title']).strip()
                if not title:
                    # Fallback to filename without extension
                    title = Path(file_path).stem
                
                # Extract author
                author = None
                if '/Author' in metadata_dict:
                    author = str(metadata_dict['/Author']).strip()
                
                # Extract creation date
                creation_date = None
                if '/CreationDate' in metadata_dict:
                    try:
                        # PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'
                        date_str = str(metadata_dict['/CreationDate'])
                        if date_str.startswith('D:'):
                            date_str = date_str[2:]
                        # Extract just the date part (YYYYMMDD)
                        if len(date_str) >= 8:
                            year = date_str[:4]
                            month = date_str[4:6]
                            day = date_str[6:8]
                            creation_date = f"{year}-{month}-{day}"
                    except (ValueError, IndexError):
                        # If date parsing fails, leave as None
                        pass
                
                # Extract format version
                format_version = None
                if hasattr(reader, 'pdf_header'):
                    format_version = reader.pdf_header
                
                return DocumentMetadata(
                    title=title,
                    author=author,
                    creation_date=creation_date,
                    file_size=file_size,
                    page_count=page_count,
                    format_version=format_version,
                    confidence_score=1.0
                )
                
        except Exception as e:
            # Return basic metadata if extraction fails
            return DocumentMetadata(
                title=Path(file_path).stem,
                file_size=self._get_file_size(file_path),
                page_count=0,
                confidence_score=0.5
            )
    
    def chunk_content(self, content: str, document_id: str, file_path: str) -> List[ContentChunk]:
        """
        Split PDF content into chunks with page-level source tracking.
        
        Args:
            content: Processed content to chunk
            document_id: Unique identifier for the document
            file_path: Original file path for source location
            
        Returns:
            List of ContentChunk objects with page information
        """
        chunks = []
        chunk_index = 0
        
        # Split content by page markers
        pages = self._split_by_pages(content)
        
        for page_num, page_content in pages.items():
            if not page_content.strip():
                continue
            
            # Split page content into chunks
            page_chunks = self._create_text_chunks(
                page_content, 
                self.config.chunk_size, 
                self.config.chunk_overlap
            )
            
            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue
                
                chunk_id = self._create_chunk_id(document_id, chunk_index)
                
                # Create source location with page information
                source_location = SourceLocation(
                    file_path=file_path,
                    page_number=page_num
                )
                
                chunk = ContentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=chunk_text.strip(),
                    content_type=ContentType.TEXT,
                    source_location=source_location,
                    metadata={
                        'page_number': page_num,
                        'chunk_index': chunk_index,
                        'character_count': len(chunk_text),
                        'word_count': len(chunk_text.split())
                    },
                    confidence_score=1.0
                )
                
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def get_content_type(self) -> ContentType:
        """Get the content type handled by this processor."""
        return ContentType.PDF
    
    def process_content(self, raw_content: str) -> str:
        """
        Process raw PDF content by cleaning and normalizing text.
        
        Args:
            raw_content: Raw extracted content
            
        Returns:
            Processed content ready for chunking
        """
        # Remove page markers for processing
        content = re.sub(r'\[PAGE \d+\]\n', '', raw_content)
        
        # Clean up common PDF extraction artifacts
        content = self._clean_pdf_text(content)
        
        return content.strip()
    
    def _split_by_pages(self, content: str) -> dict:
        """
        Split content by page markers and return page-indexed content.
        
        Args:
            content: Content with page markers
            
        Returns:
            Dictionary mapping page numbers to content
        """
        pages = {}
        current_page = 1
        current_content = []
        
        lines = content.split('\n')
        
        for line in lines:
            page_match = re.match(r'\[PAGE (\d+)\]', line)
            if page_match:
                # Save previous page content
                if current_content:
                    pages[current_page] = '\n'.join(current_content)
                
                # Start new page
                current_page = int(page_match.group(1))
                current_content = []
            else:
                current_content.append(line)
        
        # Save last page
        if current_content:
            pages[current_page] = '\n'.join(current_content)
        
        return pages
    
    def _create_text_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Create overlapping text chunks from content.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at word boundary
            if end < len(text):
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def _clean_pdf_text(self, text: str) -> str:
        """
        Clean common PDF text extraction artifacts.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\^\&\*\+\=\<\>\~\`]', ' ', text)
        
        # Fix common spacing issues
        text = re.sub(r'\s+([\.!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([\.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentence end
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
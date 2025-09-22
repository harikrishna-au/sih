"""
DOCX content extraction utilities.

Provides functionality for extracting text, metadata, and structured content
from DOCX documents with proper formatting preservation.
"""

import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from docx import Document
    from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
    from docx.oxml.ns import qn
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

from ..base import ContentExtractionError
from ...models import DocumentMetadata, ContentType

logger = logging.getLogger(__name__)


class DOCXExtractor:
    """Handles extraction of content and metadata from DOCX documents."""
    
    def __init__(self):
        """Initialize DOCX extractor."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx>=0.8.11")
    
    def extract_content(self, file_path: str) -> str:
        """
        Extract structured content from the DOCX document.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted content with structure markers
            
        Raises:
            ContentExtractionError: If content extraction fails
        """
        try:
            doc = Document(file_path)
            content_parts = []
            
            # Extract paragraphs and tables in document order
            for element in doc.element.body:
                if element.tag.endswith('p'):  # Paragraph
                    # Find corresponding paragraph object
                    for i, paragraph in enumerate(doc.paragraphs):
                        if paragraph.element == element:
                            para_content = self._extract_paragraph_content(paragraph, i)
                            if para_content.strip():
                                content_parts.append(para_content)
                            break
                
                elif element.tag.endswith('tbl'):  # Table
                    # Find corresponding table object
                    for i, table in enumerate(doc.tables):
                        if table.element == element:
                            table_content = self._extract_table_content(table, i)
                            if table_content.strip():
                                content_parts.append(table_content)
                            break
            
            # Join all content
            full_content = "\n\n".join(content_parts)
            
            # Clean and normalize
            cleaned_content = self._clean_docx_text(full_content)
            
            if not cleaned_content.strip():
                logger.warning(f"No content extracted from {file_path}")
                return ""
            
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Content extraction failed for {file_path}: {e}")
            raise ContentExtractionError(
                f"Failed to extract content from DOCX file {file_path}: {str(e)}",
                file_path=file_path,
                cause=e
            )
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract metadata from the DOCX document.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            DocumentMetadata object with extracted metadata
            
        Raises:
            ContentExtractionError: If metadata extraction fails
        """
        try:
            doc = Document(file_path)
            
            # Extract core properties
            core_props = doc.core_properties
            
            # Basic metadata
            metadata = DocumentMetadata(
                title=getattr(core_props, 'title', None) or "",
                author=getattr(core_props, 'author', None) or "",
                subject=getattr(core_props, 'subject', None) or "",
                creator=getattr(core_props, 'author', None) or "",
                producer="python-docx",
                creation_date=getattr(core_props, 'created', None),
                modification_date=getattr(core_props, 'modified', None),
                content_type=ContentType.TEXT,
                file_path=file_path,
                file_size=0,  # Will be set by caller
                page_count=len(doc.sections),  # Approximate
                language=getattr(core_props, 'language', None) or "unknown"
            )
            
            # Additional DOCX-specific metadata
            additional_metadata = {
                'paragraph_count': len(doc.paragraphs),
                'table_count': len(doc.tables),
                'section_count': len(doc.sections),
                'docx_version': getattr(core_props, 'version', None),
                'keywords': getattr(core_props, 'keywords', None),
                'category': getattr(core_props, 'category', None),
                'comments': getattr(core_props, 'comments', None),
                'last_modified_by': getattr(core_props, 'last_modified_by', None),
                'revision': getattr(core_props, 'revision', None)
            }
            
            # Filter out None values
            additional_metadata = {k: v for k, v in additional_metadata.items() if v is not None}
            metadata.additional_metadata = additional_metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction failed for {file_path}: {e}")
            raise ContentExtractionError(
                f"Failed to extract metadata from DOCX file {file_path}: {str(e)}",
                file_path=file_path,
                cause=e
            )
    
    def _extract_paragraph_content(self, paragraph, paragraph_index: int) -> str:
        """
        Extract content from a paragraph with formatting information.
        
        Args:
            paragraph: DOCX paragraph object
            paragraph_index: Index of the paragraph
            
        Returns:
            Formatted paragraph content
        """
        try:
            text = paragraph.text.strip()
            
            if not text:
                return ""
            
            # Check if it's a heading
            if paragraph.style.name.startswith('Heading'):
                level = paragraph.style.name.replace('Heading ', '')
                return f"[HEADING {level}] {text}"
            
            # Check for special formatting
            if paragraph.style.name == 'Title':
                return f"[TITLE] {text}"
            elif paragraph.style.name == 'Subtitle':
                return f"[SUBTITLE] {text}"
            elif 'Caption' in paragraph.style.name:
                return f"[CAPTION] {text}"
            elif 'Quote' in paragraph.style.name:
                return f"[QUOTE] {text}"
            
            # Regular paragraph
            return f"[PARAGRAPH {paragraph_index}]\n{text}"
            
        except Exception as e:
            logger.warning(f"Error extracting paragraph {paragraph_index}: {e}")
            return f"[PARAGRAPH {paragraph_index}]\n{paragraph.text}"
    
    def _extract_table_content(self, table, table_index: int) -> str:
        """
        Extract content from a table with structure preservation.
        
        Args:
            table: DOCX table object
            table_index: Index of the table
            
        Returns:
            Formatted table content
        """
        try:
            table_content = [f"[TABLE {table_index}]"]
            
            for row_idx, row in enumerate(table.rows):
                row_cells = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    row_cells.append(cell_text)
                
                # Join cells with tab separator
                row_content = "\t".join(row_cells)
                if row_content.strip():
                    table_content.append(f"Row {row_idx}: {row_content}")
            
            return "\n".join(table_content)
            
        except Exception as e:
            logger.warning(f"Error extracting table {table_index}: {e}")
            return f"[TABLE {table_index}]\n[Error extracting table content]"
    
    def _clean_docx_text(self, text: str) -> str:
        """
        Clean common DOCX text extraction artifacts.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Clean up common artifacts
        text = re.sub(r'\x0c', '', text)  # Form feed characters
        text = re.sub(r'\x0b', '', text)  # Vertical tab characters
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def get_document_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Get structural information about the document.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dictionary with document structure information
        """
        try:
            doc = Document(file_path)
            
            structure = {
                'paragraphs': len(doc.paragraphs),
                'tables': len(doc.tables),
                'sections': len(doc.sections),
                'headings': [],
                'styles_used': set()
            }
            
            # Analyze paragraph styles and headings
            for i, paragraph in enumerate(doc.paragraphs):
                style_name = paragraph.style.name
                structure['styles_used'].add(style_name)
                
                if style_name.startswith('Heading'):
                    level = style_name.replace('Heading ', '')
                    structure['headings'].append({
                        'level': level,
                        'text': paragraph.text[:100],  # First 100 chars
                        'paragraph_index': i
                    })
            
            structure['styles_used'] = list(structure['styles_used'])
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing document structure for {file_path}: {e}")
            return {'error': str(e)}
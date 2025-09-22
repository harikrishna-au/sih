"""
Core data models for the multimodal RAG system.

This module defines the fundamental data structures used throughout
the system for representing documents, content chunks, retrieval results,
and responses with citations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class ContentType(Enum):
    """Enumeration of supported content types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    PDF = "pdf"
    DOCX = "docx"


@dataclass
class SourceLocation:
    """
    Represents the location of content within its source file.
    
    Supports different location types:
    - Text documents: page_number, paragraph_index
    - Audio files: timestamp_start, timestamp_end
    - Images: image_coordinates (x, y, width, height)
    """
    file_path: str
    page_number: Optional[int] = None
    paragraph_index: Optional[int] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    image_coordinates: Optional[Tuple[int, int, int, int]] = None


@dataclass
class DocumentMetadata:
    """Metadata extracted from source documents."""
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[str] = None
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    duration: Optional[float] = None  # For audio files
    dimensions: Optional[Tuple[int, int]] = None  # For images
    format_version: Optional[str] = None
    language: Optional[str] = None
    confidence_score: float = 1.0


@dataclass
class ContentChunk:
    """
    A chunk of processed content with its metadata and location information.
    
    This is the fundamental unit of content that gets embedded and indexed.
    """
    chunk_id: str
    document_id: str
    content: str
    content_type: ContentType
    source_location: SourceLocation
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0


@dataclass
class DocumentContent:
    """
    Complete representation of a processed document.
    
    Contains the original content, processed content, metadata,
    and all generated chunks.
    """
    document_id: str
    content_type: ContentType
    raw_content: str
    processed_content: str
    metadata: DocumentMetadata
    chunks: List[ContentChunk] = field(default_factory=list)
    file_path: str = ""


@dataclass
class RetrievalResult:
    """
    Result from semantic search containing content and relevance information.
    
    Includes preview data for frontend display and precise source location
    for citation generation.
    """
    chunk_id: str
    content: str
    similarity_score: float
    source_location: SourceLocation
    content_type: ContentType
    preview_data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0


@dataclass
class Citation:
    """
    Citation information linking generated responses to source content.
    
    Provides numbered citations with precise location information
    and relevance scoring.
    """
    citation_id: int
    source_file: str
    location: SourceLocation
    excerpt: str
    relevance_score: float
    content_type: ContentType
    preview_data: Optional[bytes] = None


@dataclass
class GroundedResponse:
    """
    LLM-generated response with citations and supporting evidence.
    
    Contains the generated text, numbered citations, confidence scoring,
    and the original retrieval results used for generation.
    """
    response_text: str
    citations: List[Citation]
    confidence_score: float
    retrieval_results: List[RetrievalResult]
    query: str = ""
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of file validation operations."""
    is_valid: bool
    error_message: Optional[str] = None
    file_format: Optional[str] = None
    file_size: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """Result of document processing operations."""
    success: bool
    document_content: Optional[DocumentContent] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    chunks_created: int = 0


@dataclass
class SearchResult:
    """Internal search result from vector index."""
    vector_id: int
    similarity_score: float
    metadata: Dict[str, Any]


@dataclass
class BatchProcessingStatus:
    """Status information for batch processing operations."""
    total_files: int
    processed_files: int
    failed_files: int
    current_file: Optional[str] = None
    start_time: Optional[float] = None
    estimated_completion: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    status: Optional[str] = None
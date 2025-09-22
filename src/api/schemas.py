"""
Pydantic schemas for API request and response models.

Defines the data structures for API endpoints including validation,
serialization, and documentation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import time

from ..models import ContentType, SourceLocation, RetrievalResult, Citation


class ProcessingStatus(str, Enum):
    """Status enumeration for processing operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentUploadResponse(BaseModel):
    """Response model for single document upload."""
    job_id: str = Field(..., description="Unique identifier for the processing job")
    filename: str = Field(..., description="Name of the uploaded file")
    file_size: int = Field(..., description="Size of the file in bytes")
    status: ProcessingStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Status message")
    estimated_completion_time: Optional[float] = Field(None, description="Estimated completion timestamp")


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing operations."""
    job_id: str = Field(..., description="Unique identifier for the batch job")
    total_files: int = Field(..., description="Total number of files to process")
    status: ProcessingStatus = Field(..., description="Current batch processing status")
    message: str = Field(..., description="Status message")
    estimated_completion_time: Optional[float] = Field(None, description="Estimated completion timestamp")


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status queries."""
    job_id: str = Field(..., description="Job identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    total_files: int = Field(..., description="Total number of files")
    processed_files: int = Field(..., description="Number of files processed")
    failed_files: int = Field(..., description="Number of files that failed")
    current_file: Optional[str] = Field(None, description="Currently processing file")
    start_time: Optional[float] = Field(None, description="Job start timestamp")
    completion_time: Optional[float] = Field(None, description="Job completion timestamp")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    progress_percentage: float = Field(..., description="Processing progress as percentage")


class SourceLocationSchema(BaseModel):
    """Schema for source location information."""
    file_path: str = Field(..., description="Path to the source file")
    page_number: Optional[int] = Field(None, description="Page number for text documents")
    paragraph_index: Optional[int] = Field(None, description="Paragraph index within page")
    timestamp_start: Optional[float] = Field(None, description="Start timestamp for audio content")
    timestamp_end: Optional[float] = Field(None, description="End timestamp for audio content")
    image_coordinates: Optional[List[int]] = Field(None, description="Image coordinates [x, y, width, height]")


class RetrievalResultSchema(BaseModel):
    """Schema for search result items."""
    chunk_id: str = Field(..., description="Unique identifier for the content chunk")
    content: str = Field(..., description="Retrieved content text")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    relevance_score: float = Field(..., description="Relevance score after reranking")
    content_type: ContentType = Field(..., description="Type of content")
    source_location: SourceLocationSchema = Field(..., description="Source location information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    preview_available: bool = Field(False, description="Whether preview data is available")


class SearchRequest(BaseModel):
    """Request model for semantic search operations."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    k: int = Field(10, ge=1, le=100, description="Number of results to return")
    content_types: Optional[List[ContentType]] = Field(None, description="Filter by content types")
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    include_preview: bool = Field(False, description="Include preview data in results")
    rerank_results: bool = Field(True, description="Apply result reranking")
    
    @field_validator('content_types')
    @classmethod
    def validate_content_types(cls, v):
        if v is not None and len(v) == 0:
            return None
        return v


class SearchResponse(BaseModel):
    """Response model for search operations."""
    query: str = Field(..., description="Original search query")
    results: List[RetrievalResultSchema] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_time: float = Field(..., description="Search execution time in seconds")
    reranked: bool = Field(False, description="Whether results were reranked")


class CitationSchema(BaseModel):
    """Schema for citation information."""
    citation_id: int = Field(..., description="Citation number")
    source_file: str = Field(..., description="Source file path")
    location: SourceLocationSchema = Field(..., description="Location within source")
    excerpt: str = Field(..., description="Relevant excerpt from source")
    relevance_score: float = Field(..., description="Relevance score for this citation")
    content_type: ContentType = Field(..., description="Type of cited content")


class QuestionRequest(BaseModel):
    """Request model for question answering."""
    question: str = Field(..., min_length=1, max_length=1000, description="Question to answer")
    max_context_length: int = Field(4000, ge=500, le=8000, description="Maximum context length for LLM")
    max_response_length: int = Field(500, ge=50, le=2000, description="Maximum response length")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature parameter")
    include_citations: bool = Field(True, description="Include citations in response")
    search_k: int = Field(10, ge=1, le=50, description="Number of documents to retrieve for context")
    content_types: Optional[List[ContentType]] = Field(None, description="Filter context by content types")


class QuestionResponse(BaseModel):
    """Response model for question answering."""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    citations: List[CitationSchema] = Field(default_factory=list, description="Supporting citations")
    confidence_score: float = Field(..., description="Confidence score for the answer")
    generation_time: float = Field(..., description="Response generation time in seconds")
    context_used: int = Field(..., description="Number of context documents used")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="Information about the model used")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp")


class FileValidationResponse(BaseModel):
    """Response model for file validation."""
    filename: str = Field(..., description="Name of the validated file")
    is_valid: bool = Field(..., description="Whether the file is valid")
    file_format: Optional[str] = Field(None, description="Detected file format")
    file_size: int = Field(..., description="File size in bytes")
    error_message: Optional[str] = Field(None, description="Validation error message")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class SystemStatusResponse(BaseModel):
    """Response model for system status information."""
    status: str = Field(..., description="Overall system status")
    components: Dict[str, str] = Field(..., description="Status of individual components")
    active_jobs: int = Field(..., description="Number of active processing jobs")
    total_documents: int = Field(..., description="Total number of indexed documents")
    index_size: int = Field(..., description="Size of vector index")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="System version")


# Request/Response models for streaming endpoints
class StreamingResponse(BaseModel):
    """Base model for streaming responses."""
    type: str = Field(..., description="Response type")
    data: Dict[str, Any] = Field(..., description="Response data")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")


class ProgressUpdate(StreamingResponse):
    """Progress update for long-running operations."""
    type: str = Field(default="progress", description="Response type")
    job_id: str = Field(..., description="Job identifier")
    progress: float = Field(..., description="Progress percentage")
    current_operation: str = Field(..., description="Current operation description")
    estimated_remaining: Optional[float] = Field(None, description="Estimated remaining time")


class GenerationChunk(StreamingResponse):
    """Chunk of generated text for streaming responses."""
    type: str = Field(default="generation", description="Response type")
    chunk: str = Field(..., description="Generated text chunk")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    citations: List[CitationSchema] = Field(default_factory=list, description="Citations for this chunk")
"""
Citation management system for tracking and validating source-to-response mappings.

This module implements the CitationManager class that handles citation tracking,
validation, and formatting for different content types in the multimodal RAG system.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..models import (
    Citation, RetrievalResult, GroundedResponse, ContentType, 
    SourceLocation, ContentChunk
)
from .preview_generator import PreviewGenerator, PreviewConfig

logger = logging.getLogger(__name__)


@dataclass
class CitationValidationResult:
    """Result of citation validation operations."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    citation_count: int
    unique_sources: int


class CitationManager:
    """
    Manages citation tracking, validation, and formatting for grounded responses.
    
    This class provides functionality to:
    - Track source-to-response mappings
    - Validate citation accuracy and completeness
    - Format citations for different content types
    - Ensure citation consistency across responses
    """
    
    def __init__(self, preview_config: Optional[PreviewConfig] = None):
        """Initialize the citation manager."""
        self._citation_mappings: Dict[str, List[Citation]] = {}
        self._source_registry: Dict[str, Set[str]] = {}  # file_path -> set of chunk_ids
        self._citation_counter = 0
        self.preview_generator = PreviewGenerator(preview_config)
        
    def create_citations_from_results(
        self, 
        retrieval_results: List[RetrievalResult],
        response_text: str,
        min_relevance_threshold: float = 0.1
    ) -> List[Citation]:
        """
        Create citations from retrieval results for a generated response.
        
        Args:
            retrieval_results: List of retrieval results to create citations from
            response_text: The generated response text
            min_relevance_threshold: Minimum relevance score to include citation
            
        Returns:
            List of Citation objects with proper numbering and formatting
        """
        citations = []
        
        # Filter results by relevance threshold
        relevant_results = [
            result for result in retrieval_results 
            if result.relevance_score >= min_relevance_threshold
        ]
        
        # Sort by relevance score (highest first)
        relevant_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        for i, result in enumerate(relevant_results, 1):
            # Generate preview data if not already present
            preview_data = result.preview_data
            if preview_data is None:
                preview_data = self.preview_generator.generate_preview(result)
            
            citation = Citation(
                citation_id=i,
                source_file=result.source_location.file_path,
                location=result.source_location,
                excerpt=self._create_excerpt(result.content),
                relevance_score=result.relevance_score,
                content_type=result.content_type,
                preview_data=preview_data
            )
            citations.append(citation)
            
        return citations
    
    def track_citation_mapping(self, response_id: str, citations: List[Citation]) -> None:
        """
        Track the mapping between a response and its citations.
        
        Args:
            response_id: Unique identifier for the response
            citations: List of citations used in the response
        """
        self._citation_mappings[response_id] = citations
        
        # Update source registry
        for citation in citations:
            file_path = citation.source_file
            if file_path not in self._source_registry:
                self._source_registry[file_path] = set()
            # Use citation_id as a proxy for chunk_id tracking
            self._source_registry[file_path].add(str(citation.citation_id))
    
    def validate_citations(
        self, 
        citations: List[Citation],
        response_text: str
    ) -> CitationValidationResult:
        """
        Validate citations for accuracy and completeness.
        
        Args:
            citations: List of citations to validate
            response_text: The response text that should reference the citations
            
        Returns:
            CitationValidationResult with validation details
        """
        errors = []
        warnings = []
        
        # Check for duplicate citation IDs
        citation_ids = [c.citation_id for c in citations]
        if len(citation_ids) != len(set(citation_ids)):
            errors.append("Duplicate citation IDs found")
        
        # Check for missing files
        for citation in citations:
            if not Path(citation.source_file).exists():
                errors.append(f"Source file not found: {citation.source_file}")
        
        # Check for empty excerpts
        empty_excerpts = [c for c in citations if not c.excerpt.strip()]
        if empty_excerpts:
            warnings.append(f"Found {len(empty_excerpts)} citations with empty excerpts")
        
        # Check for low relevance scores
        low_relevance = [c for c in citations if c.relevance_score < 0.1]
        if low_relevance:
            warnings.append(f"Found {len(low_relevance)} citations with low relevance scores")
        
        # Check citation numbering consistency
        expected_ids = list(range(1, len(citations) + 1))
        actual_ids = sorted(citation_ids)
        if actual_ids != expected_ids:
            errors.append("Citation IDs are not consecutive starting from 1")
        
        # Count unique sources
        unique_sources = len(set(c.source_file for c in citations))
        
        return CitationValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            citation_count=len(citations),
            unique_sources=unique_sources
        )
    
    def format_citation(self, citation: Citation, format_style: str = "numbered") -> str:
        """
        Format a citation for display based on content type and style.
        
        Args:
            citation: Citation to format
            format_style: Style of formatting ("numbered", "apa", "mla")
            
        Returns:
            Formatted citation string
        """
        if format_style == "numbered":
            return self._format_numbered_citation(citation)
        elif format_style == "apa":
            return self._format_apa_citation(citation)
        elif format_style == "mla":
            return self._format_mla_citation(citation)
        else:
            return self._format_numbered_citation(citation)
    
    def _format_numbered_citation(self, citation: Citation) -> str:
        """Format citation in numbered style."""
        base_info = f"[{citation.citation_id}] {Path(citation.source_file).name}"
        
        if citation.content_type == ContentType.PDF:
            if citation.location.page_number:
                base_info += f", page {citation.location.page_number}"
        elif citation.content_type == ContentType.AUDIO:
            if citation.location.timestamp_start:
                base_info += f", {self._format_timestamp(citation.location.timestamp_start)}"
        elif citation.content_type == ContentType.IMAGE:
            if citation.location.image_coordinates:
                coords = citation.location.image_coordinates
                base_info += f", region ({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})"
        
        return base_info
    
    def _format_apa_citation(self, citation: Citation) -> str:
        """Format citation in APA style."""
        filename = Path(citation.source_file).stem
        
        if citation.content_type == ContentType.PDF:
            page_info = f", p. {citation.location.page_number}" if citation.location.page_number else ""
            return f"({filename}{page_info})"
        elif citation.content_type == ContentType.AUDIO:
            time_info = f", {self._format_timestamp(citation.location.timestamp_start)}" if citation.location.timestamp_start else ""
            return f"({filename}{time_info})"
        else:
            return f"({filename})"
    
    def _format_mla_citation(self, citation: Citation) -> str:
        """Format citation in MLA style."""
        filename = Path(citation.source_file).stem
        
        if citation.content_type == ContentType.PDF:
            page_info = f" {citation.location.page_number}" if citation.location.page_number else ""
            return f"({filename}{page_info})"
        elif citation.content_type == ContentType.AUDIO:
            time_info = f" {self._format_timestamp(citation.location.timestamp_start)}" if citation.location.timestamp_start else ""
            return f"({filename}{time_info})"
        else:
            return f"({filename})"
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp in MM:SS format."""
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _create_excerpt(self, content: str, max_length: int = 150) -> str:
        """
        Create a concise excerpt from content for citation display.
        
        Args:
            content: Full content text
            max_length: Maximum length of excerpt
            
        Returns:
            Truncated excerpt with ellipsis if needed
        """
        if len(content) <= max_length:
            return content.strip()
        
        # Find a good breaking point near the max length
        truncated = content[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # If we can break at a word boundary
            return truncated[:last_space].strip() + "..."
        else:
            return truncated.strip() + "..."
    
    def get_citation_statistics(self) -> Dict[str, int]:
        """
        Get statistics about tracked citations.
        
        Returns:
            Dictionary with citation statistics
        """
        total_responses = len(self._citation_mappings)
        total_citations = sum(len(citations) for citations in self._citation_mappings.values())
        unique_sources = len(self._source_registry)
        
        return {
            "total_responses": total_responses,
            "total_citations": total_citations,
            "unique_sources": unique_sources,
            "avg_citations_per_response": total_citations / max(total_responses, 1)
        }
    
    def generate_preview_for_result(self, result: RetrievalResult, force_regenerate: bool = False) -> Optional[bytes]:
        """
        Generate preview data for a specific retrieval result.
        
        Args:
            result: The retrieval result to generate preview for
            force_regenerate: Whether to bypass cache and regenerate
            
        Returns:
            Preview data as bytes, or None if generation fails
        """
        return self.preview_generator.generate_preview(result, force_regenerate)
    
    def get_preview_cache_stats(self) -> Dict[str, Any]:
        """Get preview cache statistics."""
        return self.preview_generator.get_cache_stats()
    
    def clear_preview_cache(self) -> None:
        """Clear the preview cache."""
        self.preview_generator.clear_cache()
        logger.info("Preview cache cleared")
    
    def clear_mappings(self) -> None:
        """Clear all citation mappings and reset counters."""
        self._citation_mappings.clear()
        self._source_registry.clear()
        self._citation_counter = 0
        logger.info("Citation mappings cleared")
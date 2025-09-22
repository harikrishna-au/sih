"""
Response generator for grounded LLM responses with citations.

This module implements the ResponseGenerator class that combines retrieval results
with LLM responses, handles citation insertion and numbering, and provides
response validation and quality scoring.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..models import (
    RetrievalResult, 
    GroundedResponse, 
    Citation, 
    ContentType,
    SourceLocation
)
from .llm_engine import LLMEngine
from ..config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class ResponseMetadata:
    """Metadata for response generation process."""
    generation_time: float
    context_length: int
    num_sources: int
    citations_found: int
    quality_score: float
    model_info: Dict[str, Any]


class CitationManager:
    """Manages citation tracking, validation, and formatting."""
    
    def __init__(self):
        self.citation_pattern = re.compile(r'\[(\d+)\]')
        self.reset()
    
    def reset(self) -> None:
        """Reset citation tracking for new response."""
        self.citations: Dict[int, Citation] = {}
        self.next_citation_id = 1
        self.source_to_citation: Dict[str, int] = {}
    
    def create_citation(
        self, 
        retrieval_result: RetrievalResult,
        excerpt_length: int = 200
    ) -> Citation:
        """
        Create a citation from a retrieval result.
        
        Args:
            retrieval_result: Source retrieval result
            excerpt_length: Maximum length of excerpt text
            
        Returns:
            Citation object with formatted information
        """
        # Create excerpt from content
        content = retrieval_result.content.strip()
        if len(content) > excerpt_length:
            excerpt = content[:excerpt_length].rsplit(' ', 1)[0] + "..."
        else:
            excerpt = content
        
        citation = Citation(
            citation_id=self.next_citation_id,
            source_file=retrieval_result.source_location.file_path,
            location=retrieval_result.source_location,
            excerpt=excerpt,
            relevance_score=retrieval_result.similarity_score,
            content_type=retrieval_result.content_type,
            preview_data=retrieval_result.preview_data
        )
        
        return citation
    
    def add_citation(self, retrieval_result: RetrievalResult) -> int:
        """
        Add a citation for a retrieval result.
        
        Args:
            retrieval_result: Source retrieval result
            
        Returns:
            Citation ID number
        """
        # Check if we already have a citation for this source
        source_key = f"{retrieval_result.source_location.file_path}:{retrieval_result.chunk_id}"
        
        if source_key in self.source_to_citation:
            return self.source_to_citation[source_key]
        
        # Create new citation
        citation = self.create_citation(retrieval_result)
        citation_id = self.next_citation_id
        
        self.citations[citation_id] = citation
        self.source_to_citation[source_key] = citation_id
        self.next_citation_id += 1
        
        return citation_id
    
    def get_citations_list(self) -> List[Citation]:
        """Get ordered list of citations."""
        return [self.citations[i] for i in sorted(self.citations.keys())]
    
    def validate_citations_in_response(self, response: str) -> Tuple[bool, List[str]]:
        """
        Validate that all citations in response are valid.
        
        Args:
            response: Generated response text
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        citation_matches = self.citation_pattern.findall(response)
        
        for match in citation_matches:
            citation_id = int(match)
            if citation_id not in self.citations:
                errors.append(f"Citation [{citation_id}] not found in source citations")
        
        # Check for unused citations
        used_citations = set(int(match) for match in citation_matches)
        available_citations = set(self.citations.keys())
        unused_citations = available_citations - used_citations
        
        if unused_citations:
            logger.warning(f"Unused citations: {unused_citations}")
        
        return len(errors) == 0, errors


class ResponseGenerator:
    """
    Generates grounded responses with citations from retrieval results.
    
    Combines retrieval results with LLM responses, handles citation insertion
    and numbering, and provides response validation and quality scoring.
    """
    
    def __init__(self, llm_engine: LLMEngine, config: Optional[LLMConfig] = None):
        """
        Initialize response generator.
        
        Args:
            llm_engine: LLM engine for response generation
            config: LLM configuration (uses engine config if None)
        """
        self.llm_engine = llm_engine
        self.config = config or llm_engine.config
        self.citation_manager = CitationManager()
        
        logger.info("ResponseGenerator initialized successfully")
    
    def generate_grounded_response(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        include_metadata: bool = True
    ) -> GroundedResponse:
        """
        Generate a grounded response with citations from retrieval results.
        
        Args:
            query: User query to answer
            retrieval_results: List of relevant retrieval results
            max_tokens: Maximum tokens for response generation
            temperature: Sampling temperature for generation
            include_metadata: Whether to include generation metadata
            
        Returns:
            GroundedResponse with text, citations, and metadata
        """
        import time
        start_time = time.time()
        
        # Reset citation manager for new response
        self.citation_manager.reset()
        
        # Prepare context with citations
        context_parts = []
        for result in retrieval_results:
            citation_id = self.citation_manager.add_citation(result)
            context_parts.append(f"[{citation_id}] {result.content}")
        
        # Generate response using LLM
        try:
            response_text = self.llm_engine.generate_with_context(
                context=context_parts,
                query=query,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Fallback response
            response_text = f"I apologize, but I encountered an error while generating a response to your query: {query}"
        
        # Validate citations in response
        citations_valid, citation_errors = self.citation_manager.validate_citations_in_response(response_text)
        if not citations_valid:
            logger.warning(f"Citation validation errors: {citation_errors}")
        
        # Calculate quality score
        quality_score = self._calculate_response_quality(
            response_text, context_parts, query, retrieval_results
        )
        
        # Get citations list
        citations = self.citation_manager.get_citations_list()
        
        # Create metadata
        generation_metadata = {}
        if include_metadata:
            generation_time = time.time() - start_time
            generation_metadata = {
                "generation_time": generation_time,
                "context_length": sum(len(part) for part in context_parts),
                "num_sources": len(retrieval_results),
                "citations_found": len(citations),
                "quality_score": quality_score,
                "citations_valid": citations_valid,
                "citation_errors": citation_errors,
                "model_info": self.llm_engine.get_model_info() if self.llm_engine.is_model_loaded() else {}
            }
        
        return GroundedResponse(
            response_text=response_text,
            citations=citations,
            confidence_score=quality_score,
            retrieval_results=retrieval_results,
            query=query,
            generation_metadata=generation_metadata
        )
    
    def format_response_with_citations(
        self, 
        grounded_response: GroundedResponse,
        citation_format: str = "numbered"
    ) -> str:
        """
        Format a grounded response with properly formatted citations.
        
        Args:
            grounded_response: Response to format
            citation_format: Citation format style ("numbered", "footnote", "inline")
            
        Returns:
            Formatted response string with citations
        """
        response_text = grounded_response.response_text
        citations = grounded_response.citations
        
        if citation_format == "numbered":
            return self._format_numbered_citations(response_text, citations)
        elif citation_format == "footnote":
            return self._format_footnote_citations(response_text, citations)
        elif citation_format == "inline":
            return self._format_inline_citations(response_text, citations)
        else:
            logger.warning(f"Unknown citation format: {citation_format}, using numbered")
            return self._format_numbered_citations(response_text, citations)
    
    def _format_numbered_citations(self, response_text: str, citations: List[Citation]) -> str:
        """Format response with numbered citations."""
        if not citations:
            return response_text
        
        formatted_response = response_text + "\n\n**Sources:**\n"
        for citation in citations:
            location_str = self._format_source_location(citation.location)
            formatted_response += f"[{citation.citation_id}] {citation.source_file}"
            if location_str:
                formatted_response += f" ({location_str})"
            formatted_response += f": {citation.excerpt}\n"
        
        return formatted_response
    
    def _format_footnote_citations(self, response_text: str, citations: List[Citation]) -> str:
        """Format response with footnote-style citations."""
        if not citations:
            return response_text
        
        # Replace [1] with ¹, [2] with ², etc.
        footnote_response = response_text
        for citation in citations:
            old_ref = f"[{citation.citation_id}]"
            new_ref = self._get_superscript_number(citation.citation_id)
            footnote_response = footnote_response.replace(old_ref, new_ref)
        
        # Add footnotes at bottom
        footnote_response += "\n\n**References:**\n"
        for citation in citations:
            footnote_num = self._get_superscript_number(citation.citation_id)
            location_str = self._format_source_location(citation.location)
            footnote_response += f"{footnote_num} {citation.source_file}"
            if location_str:
                footnote_response += f" ({location_str})"
            footnote_response += f": {citation.excerpt}\n"
        
        return footnote_response
    
    def _format_inline_citations(self, response_text: str, citations: List[Citation]) -> str:
        """Format response with inline citations."""
        if not citations:
            return response_text
        
        # Replace [1] with (Source: filename, page X)
        inline_response = response_text
        for citation in citations:
            old_ref = f"[{citation.citation_id}]"
            location_str = self._format_source_location(citation.location)
            source_name = citation.source_file.split('/')[-1]  # Just filename
            if location_str:
                new_ref = f"(Source: {source_name}, {location_str})"
            else:
                new_ref = f"(Source: {source_name})"
            inline_response = inline_response.replace(old_ref, new_ref)
        
        return inline_response
    
    def _format_source_location(self, location: SourceLocation) -> str:
        """Format source location information."""
        parts = []
        
        if location.page_number is not None:
            parts.append(f"page {location.page_number}")
        
        if location.paragraph_index is not None:
            parts.append(f"paragraph {location.paragraph_index}")
        
        if location.timestamp_start is not None:
            if location.timestamp_end is not None:
                parts.append(f"{location.timestamp_start:.1f}s-{location.timestamp_end:.1f}s")
            else:
                parts.append(f"{location.timestamp_start:.1f}s")
        
        if location.image_coordinates is not None:
            x, y, w, h = location.image_coordinates
            parts.append(f"region ({x},{y},{w},{h})")
        
        return ", ".join(parts)
    
    def _get_superscript_number(self, num: int) -> str:
        """Convert number to superscript Unicode characters."""
        superscript_map = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
        }
        return ''.join(superscript_map.get(digit, digit) for digit in str(num))
    
    def _calculate_response_quality(
        self,
        response: str,
        context: List[str],
        query: str,
        retrieval_results: List[RetrievalResult]
    ) -> float:
        """
        Calculate comprehensive quality score for generated response.
        
        Args:
            response: Generated response text
            context: Context used for generation
            query: Original query
            retrieval_results: Source retrieval results
            
        Returns:
            Quality score between 0 and 1
        """
        # Start with base LLM quality score
        base_quality = self.llm_engine.validate_response_quality(response, context, query)
        
        # Additional quality factors
        quality_factors = []
        
        # Citation coverage - how many sources are actually cited
        cited_numbers = set(int(match) for match in self.citation_manager.citation_pattern.findall(response))
        available_citations = len(self.citation_manager.citations)
        if available_citations > 0:
            citation_coverage = len(cited_numbers) / available_citations
            quality_factors.append(citation_coverage * 0.2)  # 20% weight
        
        # Response completeness - not too short or too long
        response_length = len(response.split())
        if 20 <= response_length <= 200:  # Reasonable length
            quality_factors.append(0.1)
        elif response_length < 10:  # Too short
            quality_factors.append(-0.2)
        elif response_length > 300:  # Too long
            quality_factors.append(-0.1)
        
        # Query relevance - check if response addresses the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        query_overlap = len(query_words.intersection(response_words)) / max(len(query_words), 1)
        quality_factors.append(query_overlap * 0.15)  # 15% weight
        
        # Source relevance - use average similarity score of retrieval results
        if retrieval_results:
            avg_similarity = sum(r.similarity_score for r in retrieval_results) / len(retrieval_results)
            quality_factors.append(avg_similarity * 0.15)  # 15% weight
        
        # Combine all factors
        final_quality = base_quality + sum(quality_factors)
        return max(0.0, min(1.0, final_quality))  # Clamp to [0, 1]
    
    def validate_response(
        self, 
        grounded_response: GroundedResponse,
        min_quality_threshold: float = 0.5
    ) -> Tuple[bool, List[str]]:
        """
        Validate a grounded response for quality and correctness.
        
        Args:
            grounded_response: Response to validate
            min_quality_threshold: Minimum acceptable quality score
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check response text
        if not grounded_response.response_text or len(grounded_response.response_text.strip()) < 10:
            issues.append("Response text is too short or empty")
        
        # Check quality score
        if grounded_response.confidence_score < min_quality_threshold:
            issues.append(f"Quality score {grounded_response.confidence_score:.2f} below threshold {min_quality_threshold}")
        
        # Check citations
        if not grounded_response.citations and grounded_response.retrieval_results:
            issues.append("No citations found despite having retrieval results")
        
        # Validate citation references in text
        citation_pattern = re.compile(r'\[(\d+)\]')
        cited_numbers = set(int(match) for match in citation_pattern.findall(grounded_response.response_text))
        available_citations = set(c.citation_id for c in grounded_response.citations)
        
        invalid_citations = cited_numbers - available_citations
        if invalid_citations:
            issues.append(f"Invalid citation references: {invalid_citations}")
        
        # Check for hallucination indicators
        hallucination_phrases = [
            "i don't know", "i'm not sure", "i cannot", "i apologize",
            "based on my knowledge", "in my opinion"
        ]
        response_lower = grounded_response.response_text.lower()
        for phrase in hallucination_phrases:
            if phrase in response_lower and not grounded_response.retrieval_results:
                issues.append(f"Potential hallucination detected: '{phrase}' without supporting sources")
        
        return len(issues) == 0, issues
    
    def improve_response_quality(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        initial_response: GroundedResponse,
        max_attempts: int = 2
    ) -> GroundedResponse:
        """
        Attempt to improve response quality through regeneration.
        
        Args:
            query: Original query
            retrieval_results: Source retrieval results
            initial_response: Initial response to improve
            max_attempts: Maximum improvement attempts
            
        Returns:
            Improved grounded response
        """
        best_response = initial_response
        
        for attempt in range(max_attempts):
            # Adjust generation parameters for improvement
            temperature = max(0.1, self.config.temperature - (attempt * 0.2))
            
            # Regenerate response
            improved_response = self.generate_grounded_response(
                query=query,
                retrieval_results=retrieval_results,
                temperature=temperature,
                include_metadata=True
            )
            
            # Keep the better response
            if improved_response.confidence_score > best_response.confidence_score:
                best_response = improved_response
                logger.info(f"Improved response quality from {initial_response.confidence_score:.2f} "
                           f"to {improved_response.confidence_score:.2f}")
            
            # Stop if we achieve good quality
            if best_response.confidence_score >= 0.8:
                break
        
        return best_response
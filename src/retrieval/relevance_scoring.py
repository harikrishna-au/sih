"""
Relevance scoring utilities for semantic search results.

Provides comprehensive relevance scoring based on multiple factors including
semantic similarity, content type preferences, recency, length, and keyword matching.
"""

import re
import logging
from typing import Dict, Any
from ..models import RetrievalResult, ContentType
from .search_config import SearchConfig

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """Calculates comprehensive relevance scores for search results."""
    
    def __init__(self, config: SearchConfig):
        """Initialize relevance scorer with configuration."""
        self.config = config
    
    def calculate_relevance_score(self, result: RetrievalResult, query: str) -> float:
        """
        Calculate comprehensive relevance score for a search result.
        
        Args:
            result: Search result to score
            query: Original search query
            
        Returns:
            Relevance score between 0 and 1
        """
        weights = self.config.relevance_weights
        
        # Base semantic similarity score
        semantic_score = result.similarity_score * weights["semantic_similarity"]
        
        # Content type boost (prefer certain content types)
        content_type_score = self._calculate_content_type_boost(result) * weights["content_type_boost"]
        
        # Recency boost (prefer newer content if metadata available)
        recency_score = self._calculate_recency_boost(result) * weights["recency_boost"]
        
        # Length penalty (penalize very short or very long content)
        length_score = self._calculate_length_penalty(result) * weights["length_penalty"]
        
        # Keyword match boost (exact keyword matches)
        keyword_score = self._calculate_keyword_match_score(result, query) * weights["keyword_match"]
        
        # Combine all scores
        total_score = (
            semantic_score + 
            content_type_score + 
            recency_score + 
            length_score + 
            keyword_score
        )
        
        # Normalize to [0, 1] range
        return min(1.0, max(0.0, total_score))
    
    def _calculate_content_type_boost(self, result: RetrievalResult) -> float:
        """Calculate content type preference boost."""
        # Boost scores based on content type preferences
        content_type_boosts = {
            ContentType.TEXT: 0.0,      # Baseline
            ContentType.IMAGE: 0.1,     # Slight boost for visual content
            ContentType.AUDIO: 0.05     # Small boost for audio content
        }
        
        return content_type_boosts.get(result.content_type, 0.0)
    
    def _calculate_recency_boost(self, result: RetrievalResult) -> float:
        """Calculate recency boost based on document metadata."""
        # Check if creation date is available in metadata
        creation_date = result.metadata.get('creation_date')
        if not creation_date:
            return 0.0
        
        try:
            # Simple recency calculation (would need proper date parsing in real implementation)
            # For now, return a small boost for documents with date metadata
            return 0.05
        except:
            return 0.0
    
    def _calculate_length_penalty(self, result: RetrievalResult) -> float:
        """Calculate penalty/boost based on content length."""
        content_length = len(result.content)
        
        # Optimal length range (adjust based on use case)
        optimal_min = 50
        optimal_max = 500
        
        if optimal_min <= content_length <= optimal_max:
            return 0.1  # Boost for optimal length
        elif content_length < optimal_min:
            return -0.05  # Small penalty for very short content
        elif content_length > optimal_max * 2:
            return -0.1  # Penalty for very long content
        else:
            return 0.0  # Neutral for moderate length
    
    def _calculate_keyword_match_score(self, result: RetrievalResult, query: str) -> float:
        """Calculate boost for exact keyword matches."""
        if not query or not result.content:
            return 0.0
        
        # Extract keywords from query (simple approach)
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        content_words = set(re.findall(r'\b\w+\b', result.content.lower()))
        
        if not query_words:
            return 0.0
        
        # Calculate exact match ratio
        exact_matches = len(query_words.intersection(content_words))
        match_ratio = exact_matches / len(query_words)
        
        # Return scaled boost
        return match_ratio * 0.2  # Max 0.2 boost for perfect keyword match
    
    def calculate_query_document_interaction(self, result: RetrievalResult, query: str) -> float:
        """Calculate query-document interaction features."""
        # Position of query terms in document
        position_score = self._calculate_position_score(result.content, query)
        
        # Query term frequency in document
        frequency_score = self._calculate_term_frequency_score(result.content, query)
        
        # Document structure features (if available)
        structure_score = self._calculate_structure_score(result)
        
        return (position_score + frequency_score + structure_score) / 3
    
    def _calculate_position_score(self, content: str, query: str) -> float:
        """Calculate score based on position of query terms in content."""
        if not query or not content:
            return 0.0
        
        query_words = re.findall(r'\b\w+\b', query.lower())
        content_lower = content.lower()
        
        position_scores = []
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1:
                # Earlier positions get higher scores
                relative_pos = pos / len(content_lower)
                position_scores.append(1.0 - relative_pos)
        
        return sum(position_scores) / len(query_words) if query_words else 0.0
    
    def _calculate_term_frequency_score(self, content: str, query: str) -> float:
        """Calculate score based on term frequency."""
        if not query or not content:
            return 0.0
        
        query_words = re.findall(r'\b\w+\b', query.lower())
        content_words = re.findall(r'\b\w+\b', content.lower())
        
        if not content_words:
            return 0.0
        
        tf_scores = []
        for word in query_words:
            tf = content_words.count(word) / len(content_words)
            tf_scores.append(tf)
        
        return sum(tf_scores) / len(query_words) if query_words else 0.0
    
    def _calculate_structure_score(self, result: RetrievalResult) -> float:
        """Calculate score based on document structure features."""
        # Check if content appears to be from title, heading, or important section
        content = result.content.strip()
        
        # Simple heuristics for structure importance
        if len(content) < 100 and content.isupper():
            return 0.3  # Likely a title or heading
        elif result.source_location.paragraph_index == 0:
            return 0.2  # First paragraph often important
        elif any(marker in content.lower() for marker in ['summary', 'conclusion', 'abstract']):
            return 0.25  # Important sections
        else:
            return 0.0
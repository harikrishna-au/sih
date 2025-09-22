"""
Result processing utilities for search results.

Provides functionality for filtering, diversity optimization, and result
post-processing operations.
"""

import logging
from typing import List, Dict, Any, Optional
from ..models import RetrievalResult
from .search_config import SearchFilter, SearchConfig

logger = logging.getLogger(__name__)


class ResultProcessor:
    """Handles post-processing of search results."""
    
    def __init__(self, config: SearchConfig):
        """Initialize result processor with configuration."""
        self.config = config
    
    def apply_search_filter(
        self, 
        results: List[RetrievalResult], 
        search_filter: SearchFilter
    ) -> List[RetrievalResult]:
        """Apply additional filtering to search results."""
        filtered_results = []
        
        for result in results:
            # Check confidence threshold
            if result.similarity_score < search_filter.confidence_threshold:
                continue
            
            # Check document ID filter
            if search_filter.document_ids and not any(
                doc_id in result.source_location.file_path for doc_id in search_filter.document_ids
            ):
                continue
            
            # Check file path filter
            if search_filter.file_paths and result.source_location.file_path not in search_filter.file_paths:
                continue
            
            # Check metadata filters
            if search_filter.metadata_filters:
                metadata_match = True
                for key, value in search_filter.metadata_filters.items():
                    if key not in result.metadata or result.metadata[key] != value:
                        metadata_match = False
                        break
                if not metadata_match:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def ensure_result_diversity(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Ensure diversity in search results by removing near-duplicates.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Filtered list with diverse results
        """
        if not results or len(results) <= 1:
            return results
        
        diverse_results = [results[0]]  # Always include the top result
        
        for result in results[1:]:
            # Check if this result is too similar to any already selected result
            is_diverse = True
            
            for selected_result in diverse_results:
                # Check content similarity (simple approach)
                if self._calculate_content_similarity(result.content, selected_result.content) > self.config.diversity_threshold:
                    is_diverse = False
                    break
                
                # Check if from same source location
                if (result.source_location.file_path == selected_result.source_location.file_path and
                    result.source_location.page_number == selected_result.source_location.page_number):
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def optimize_result_diversity(
        self, 
        results: List[RetrievalResult], 
        diversity_lambda: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Optimize result diversity using Maximal Marginal Relevance (MMR).
        
        Args:
            results: Search results to diversify
            diversity_lambda: Balance between relevance and diversity (0-1)
            
        Returns:
            Diversified list of results
        """
        if not results or len(results) <= 1:
            return results
        
        # Start with the highest relevance result
        selected = [results[0]]
        remaining = results[1:]
        
        while remaining and len(selected) < len(results):
            mmr_scores = []
            
            for candidate in remaining:
                # Relevance score
                relevance = candidate.relevance_score
                
                # Maximum similarity to already selected results
                max_similarity = 0.0
                for selected_result in selected:
                    similarity = self._calculate_content_similarity(
                        candidate.content, 
                        selected_result.content
                    )
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score: balance relevance and diversity
                mmr_score = diversity_lambda * relevance - (1 - diversity_lambda) * max_similarity
                mmr_scores.append((mmr_score, candidate))
            
            # Select candidate with highest MMR score
            if mmr_scores:
                best_candidate = max(mmr_scores, key=lambda x: x[0])[1]
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return selected
    
    def build_filter_conditions(self, search_filter: Optional[SearchFilter]) -> Optional[Dict[str, Any]]:
        """Build filter conditions for vector store query."""
        if not search_filter:
            return None
        
        conditions = {}
        
        if search_filter.document_ids:
            conditions['document_id'] = search_filter.document_ids
        
        if search_filter.file_paths:
            conditions['file_path'] = search_filter.file_paths
        
        if search_filter.metadata_filters:
            conditions.update(search_filter.metadata_filters)
        
        return conditions if conditions else None
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate simple content similarity between two text strings.
        
        Args:
            content1: First text content
            content2: Second text content
            
        Returns:
            Similarity score between 0 and 1
        """
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based similarity (could be improved with more sophisticated methods)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
"""
Search performance statistics tracking and reporting.

Provides comprehensive statistics collection and reporting for search operations,
including performance metrics, usage patterns, and optimization insights.
"""

import time
import logging
from typing import Dict, Any, List
from .search_config import SearchConfig

logger = logging.getLogger(__name__)


class SearchStatistics:
    """Tracks and manages search performance statistics."""
    
    def __init__(self):
        """Initialize search statistics tracker."""
        self._stats = {
            'total_searches': 0,
            'text_searches': 0,
            'image_searches': 0,
            'cross_modal_searches': 0,
            'average_search_time': 0.0,
            'total_results_returned': 0,
            'reranking_operations': 0,
            'average_reranking_time': 0.0,
            'relevance_score_improvements': 0
        }
    
    def update_search_stats(self, search_type: str, search_time: float, result_count: int) -> None:
        """
        Update search performance statistics.
        
        Args:
            search_type: Type of search performed ('text', 'image', 'cross_modal')
            search_time: Time taken for the search
            result_count: Number of results returned
        """
        self._stats['total_searches'] += 1
        self._stats[f'{search_type}_searches'] += 1
        self._stats['total_results_returned'] += result_count
        
        # Update average search time
        total_time = self._stats['average_search_time'] * (self._stats['total_searches'] - 1)
        self._stats['average_search_time'] = (total_time + search_time) / self._stats['total_searches']
    
    def update_reranking_stats(
        self, 
        reranking_time: float, 
        original_scores: List[float], 
        reranked_results: List[Any]
    ) -> None:
        """
        Update reranking performance statistics.
        
        Args:
            reranking_time: Time taken for reranking
            original_scores: Original relevance scores
            reranked_results: Results after reranking
        """
        self._stats['reranking_operations'] += 1
        
        # Update average reranking time
        total_ops = self._stats['reranking_operations']
        total_time = self._stats['average_reranking_time'] * (total_ops - 1)
        self._stats['average_reranking_time'] = (total_time + reranking_time) / total_ops
        
        # Calculate relevance score improvements
        new_scores = [r.relevance_score for r in reranked_results]
        if len(original_scores) == len(new_scores):
            improvements = sum(1 for old, new in zip(original_scores, new_scores) if new > old)
            self._stats['relevance_score_improvements'] += improvements
    
    def get_comprehensive_stats(self, config: SearchConfig, cache_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive search performance statistics.
        
        Args:
            config: Search configuration
            cache_stats: Cache statistics
            
        Returns:
            Dictionary containing all search metrics and performance data
        """
        return {
            **self._stats,
            **cache_stats,
            'config': {
                'similarity_threshold': config.similarity_threshold,
                'max_results': config.max_results,
                'enable_cross_modal': config.enable_cross_modal,
                'enable_reranking': config.enable_reranking,
                'diversity_threshold': config.diversity_threshold,
                'reranking_model': config.reranking_model,
                'relevance_weights': config.relevance_weights
            }
        }
    
    def reset_stats(self) -> None:
        """Reset all statistics to initial values."""
        for key in self._stats:
            if isinstance(self._stats[key], (int, float)):
                self._stats[key] = 0 if isinstance(self._stats[key], int) else 0.0
        
        logger.info("Search statistics reset")
    
    def get_performance_summary(self) -> str:
        """
        Get a formatted performance summary.
        
        Returns:
            Formatted string with key performance metrics
        """
        total_searches = self._stats['total_searches']
        if total_searches == 0:
            return "No searches performed yet."
        
        avg_results = self._stats['total_results_returned'] / total_searches
        reranking_rate = (self._stats['reranking_operations'] / total_searches) * 100
        
        summary = f"""
Search Performance Summary:
- Total searches: {total_searches}
- Average search time: {self._stats['average_search_time']:.3f}s
- Average results per search: {avg_results:.1f}
- Reranking usage: {reranking_rate:.1f}%
- Average reranking time: {self._stats['average_reranking_time']:.3f}s
- Relevance improvements: {self._stats['relevance_score_improvements']}
        """.strip()
        
        return summary
"""
Search result caching system for performance optimization.

Provides caching functionality for search queries and results with
LRU-style cache management and statistics tracking.
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional
from ..models import RetrievalResult
from .search_config import SearchFilter, SearchConfig

logger = logging.getLogger(__name__)


class SearchCache:
    """Manages caching of search results for performance optimization."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize search cache.
        
        Args:
            max_size: Maximum number of cached queries
        """
        self._cache: Dict[str, List[RetrievalResult]] = {}
        self._cache_max_size = max_size
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cached_results(
        self, 
        query: str, 
        top_k: int, 
        search_filter: Optional[SearchFilter],
        similarity_threshold: Optional[float],
        config: SearchConfig
    ) -> Optional[List[RetrievalResult]]:
        """
        Retrieve cached results for a query.
        
        Args:
            query: Search query
            top_k: Number of results requested
            search_filter: Search filter conditions
            similarity_threshold: Similarity threshold
            config: Search configuration
            
        Returns:
            Cached results if available, None otherwise
        """
        cache_key = self._generate_cache_key(query, top_k, search_filter, similarity_threshold, config)
        
        if cache_key in self._cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self._cache[cache_key][:top_k]
        
        self._cache_misses += 1
        return None
    
    def cache_results(
        self, 
        query: str, 
        top_k: int, 
        search_filter: Optional[SearchFilter],
        similarity_threshold: Optional[float],
        config: SearchConfig,
        results: List[RetrievalResult]
    ) -> None:
        """
        Cache search results.
        
        Args:
            query: Search query
            top_k: Number of results requested
            search_filter: Search filter conditions
            similarity_threshold: Similarity threshold
            config: Search configuration
            results: Results to cache
        """
        cache_key = self._generate_cache_key(query, top_k, search_filter, similarity_threshold, config)
        
        # Implement simple LRU-like cache management
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entries (simple approach)
            keys_to_remove = list(self._cache.keys())[:self._cache_max_size // 4]
            for key in keys_to_remove:
                del self._cache[key]
        
        self._cache[cache_key] = results.copy()
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        logger.info("Search result cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self._cache),
            'cache_max_size': self._cache_max_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }
    
    def _generate_cache_key(
        self, 
        query: str, 
        top_k: int, 
        search_filter: Optional[SearchFilter],
        similarity_threshold: Optional[float],
        config: SearchConfig
    ) -> str:
        """Generate cache key for query results."""
        # Create a string representation of the search parameters
        cache_data = f"{query}:{top_k}:{similarity_threshold or config.similarity_threshold}"
        
        if search_filter:
            cache_data += f":{search_filter.content_types}:{search_filter.confidence_threshold}"
        
        # Generate hash
        return hashlib.md5(cache_data.encode()).hexdigest()
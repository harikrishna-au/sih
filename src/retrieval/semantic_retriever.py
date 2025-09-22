"""
Semantic retrieval engine for multimodal content.

Implements core semantic search functionality with vector similarity search,
query embedding generation, result filtering, and cross-modal search capabilities.
Provides the main interface for retrieving relevant content from the vector store.
"""

import logging
from typing import List, Optional

from ..models import RetrievalResult, ContentChunk
from ..embeddings.unified_embedding_generator import UnifiedEmbeddingGenerator
from .vectordb.qdrant_vector_store import QdrantVectorStore
from .search_config import SearchConfig, SearchFilter
from .reranking_algorithms import ResultReranker
from .result_processing import ResultProcessor
from .search_cache import SearchCache
from .search_statistics import SearchStatistics
from .search_methods import SearchMethods

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Core semantic search functionality for multimodal content.
    
    Features:
    - Vector similarity search with configurable thresholds
    - Query embedding generation for text and image queries
    - Cross-modal search across text, image, and audio content
    - Result filtering by content type, metadata, and confidence
    - Performance optimization with caching and batch operations
    """
    
    def __init__(
        self, 
        vector_store: QdrantVectorStore,
        embedding_generator: UnifiedEmbeddingGenerator,
        config: Optional[SearchConfig] = None
    ):
        """
        Initialize semantic retriever.
        
        Args:
            vector_store: Vector database for similarity search
            embedding_generator: Unified embedding generator
            config: Search configuration (uses defaults if None)
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.config = config or SearchConfig()
        
        # Initialize components
        self.reranker = ResultReranker(self.config)
        self.result_processor = ResultProcessor(self.config)
        self.cache = SearchCache()
        self.statistics = SearchStatistics()
        self.search_methods = SearchMethods(self)
        
        logger.info("SemanticRetriever initialized successfully")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        search_filter: Optional[SearchFilter] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Perform semantic search using text query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            search_filter: Optional filter conditions
            similarity_threshold: Minimum similarity score (overrides config)
            
        Returns:
            List of retrieval results sorted by relevance
        """
        return self.search_methods.text_search(query, top_k, search_filter, similarity_threshold)
    
    def search_by_image(
        self, 
        image_path: str, 
        top_k: int = 10,
        search_filter: Optional[SearchFilter] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Perform semantic search using image query.
        
        Args:
            image_path: Path to query image
            top_k: Number of results to return
            search_filter: Optional filter conditions
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieval results sorted by relevance
        """
        return self.search_methods.image_search(image_path, top_k, search_filter, similarity_threshold)
    
    def cross_modal_search(
        self, 
        query: str, 
        top_k: int = 10,
        include_images: bool = True,
        include_audio: bool = True,
        search_filter: Optional[SearchFilter] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Perform cross-modal semantic search across all content types.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            include_images: Whether to include image results
            include_audio: Whether to include audio results
            search_filter: Optional filter conditions
            similarity_threshold: Minimum similarity score
            
        Returns:
            Combined list of retrieval results from all modalities
        """
        return self.search_methods.cross_modal_search(
            query, top_k, include_images, include_audio, search_filter, similarity_threshold
        )
    
    def search_similar_content(
        self, 
        content_chunk: ContentChunk, 
        top_k: int = 10,
        exclude_same_document: bool = True,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Find content similar to a given content chunk.
        
        Args:
            content_chunk: Reference content chunk
            top_k: Number of results to return
            exclude_same_document: Whether to exclude results from same document
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar content chunks
        """
        return self.search_methods.similar_content_search(
            content_chunk, top_k, exclude_same_document, similarity_threshold
        )
    
    def batch_search(
        self, 
        queries: List[str], 
        top_k: int = 10,
        search_filter: Optional[SearchFilter] = None
    ) -> List[List[RetrievalResult]]:
        """
        Perform batch semantic search for multiple queries efficiently.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            search_filter: Optional filter conditions
            
        Returns:
            List of result lists, one for each query
        """
        return self.search_methods.batch_search(queries, top_k, search_filter)
    
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
        return self.result_processor.optimize_result_diversity(results, diversity_lambda)
    
    def get_search_statistics(self):
        """
        Get comprehensive search performance statistics.
        
        Returns:
            Dictionary containing search metrics and performance data
        """
        cache_stats = self.cache.get_cache_stats()
        return self.statistics.get_comprehensive_stats(self.config, cache_stats)
    
    def clear_cache(self) -> None:
        """Clear the query result cache."""
        self.cache.clear_cache()
        logger.info("Search result cache cleared")
    
    def update_config(self, config: SearchConfig) -> None:
        """
        Update search configuration.
        
        Args:
            config: New search configuration
        """
        self.config = config
        # Update components with new config
        self.reranker = ResultReranker(config)
        self.result_processor = ResultProcessor(config)
        # Clear cache since configuration changed
        self.clear_cache()
        logger.info("Search configuration updated")



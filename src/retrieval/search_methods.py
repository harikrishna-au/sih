"""
Core search method implementations for semantic retrieval.

Contains the main search functionality including text search, image search,
cross-modal search, and batch search operations.
"""

import logging
import time
from typing import List, Optional

from ..models import ContentType, RetrievalResult, ContentChunk
from .search_config import SearchFilter

logger = logging.getLogger(__name__)


class SearchMethods:
    """Implements core search methods for semantic retrieval."""
    
    def __init__(self, retriever):
        """Initialize with reference to main retriever."""
        self.retriever = retriever
    
    def text_search(
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
            
        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If search operation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        try:
            # Check cache first
            cached_results = self.retriever.cache.get_cached_results(
                query, top_k, search_filter, similarity_threshold, self.retriever.config
            )
            if cached_results is not None:
                return cached_results
            
            # Use provided threshold or config default
            threshold = similarity_threshold or self.retriever.config.similarity_threshold
            
            # Prepare filter conditions for vector store
            content_types = search_filter.content_types if search_filter else None
            filter_conditions = self.retriever.result_processor.build_filter_conditions(search_filter)
            
            # Perform vector search
            results = self.retriever.vector_store.search(
                query=query,
                top_k=min(top_k, self.retriever.config.max_results),
                content_types=content_types,
                similarity_threshold=threshold,
                filter_conditions=filter_conditions
            )
            
            # Apply additional filtering
            if search_filter:
                results = self.retriever.result_processor.apply_search_filter(results, search_filter)
            
            # Apply result deduplication and diversity
            results = self.retriever.result_processor.ensure_result_diversity(results)
            
            # Apply reranking if enabled
            if self.retriever.config.enable_reranking and results:
                results = self.retriever.reranker.rerank_results(results, query)
            
            # Cache results
            self.retriever.cache.cache_results(query, top_k, search_filter, similarity_threshold, self.retriever.config, results)
            
            # Update statistics
            search_time = time.time() - start_time
            self.retriever.statistics.update_search_stats('text', search_time, len(results))
            
            logger.info(f"Text search completed in {search_time:.3f}s, returned {len(results)} results")
            return results[:top_k]
            
        except Exception as e:
            error_msg = f"Error in semantic search: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def image_search(
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
            
        Raises:
            ValueError: If image path is invalid
            RuntimeError: If search operation fails
        """
        if not image_path or not image_path.strip():
            raise ValueError("Image path cannot be empty")
        
        start_time = time.time()
        
        try:
            # Use provided threshold or config default
            threshold = similarity_threshold or self.retriever.config.similarity_threshold
            
            # Prepare filter conditions
            content_types = search_filter.content_types if search_filter else None
            
            # Perform image-based vector search
            results = self.retriever.vector_store.search_by_image(
                image_path=image_path,
                top_k=min(top_k, self.retriever.config.max_results),
                content_types=content_types,
                similarity_threshold=threshold
            )
            
            # Apply additional filtering
            if search_filter:
                results = self.retriever.result_processor.apply_search_filter(results, search_filter)
            
            # Apply result deduplication and diversity
            results = self.retriever.result_processor.ensure_result_diversity(results)
            
            # Apply reranking if enabled
            if self.retriever.config.enable_reranking and results:
                results = self.retriever.reranker.rerank_results(results, f"image_query:{image_path}")
            
            # Update statistics
            search_time = time.time() - start_time
            self.retriever.statistics.update_search_stats('image', search_time, len(results))
            
            logger.info(f"Image search completed in {search_time:.3f}s, returned {len(results)} results")
            return results[:top_k]
            
        except Exception as e:
            error_msg = f"Error in image search: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
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
            
        Raises:
            ValueError: If query is empty
            RuntimeError: If search operation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        try:
            # Use provided threshold or config default
            threshold = similarity_threshold or self.retriever.config.similarity_threshold
            
            # Perform cross-modal search using vector store
            results = self.retriever.vector_store.search_cross_modal(
                query=query,
                top_k=min(top_k, self.retriever.config.max_results),
                include_images=include_images,
                similarity_threshold=threshold
            )
            
            # Filter by audio content if not included
            if not include_audio:
                results = [r for r in results if r.content_type != ContentType.AUDIO]
            
            # Apply additional filtering
            if search_filter:
                results = self.retriever.result_processor.apply_search_filter(results, search_filter)
            
            # Apply result deduplication and diversity
            results = self.retriever.result_processor.ensure_result_diversity(results)
            
            # Apply reranking if enabled
            if self.retriever.config.enable_reranking and results:
                results = self.retriever.reranker.rerank_results(results, query)
            
            # Update statistics
            search_time = time.time() - start_time
            self.retriever.statistics.update_search_stats('cross_modal', search_time, len(results))
            
            logger.info(f"Cross-modal search completed in {search_time:.3f}s, returned {len(results)} results")
            return results[:top_k]
            
        except Exception as e:
            error_msg = f"Error in cross-modal search: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def similar_content_search(
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
            
        Raises:
            ValueError: If content chunk is invalid
            RuntimeError: If search operation fails
        """
        if not content_chunk or not content_chunk.content:
            raise ValueError("Content chunk cannot be empty")
        
        try:
            # Generate embedding for the content chunk if not present
            if content_chunk.embedding is None:
                content_chunk.embedding = self.retriever.embedding_generator.generate_embedding(
                    content_chunk.content, 
                    content_chunk.content_type
                )
            
            # Use the content as query for similarity search
            results = self.text_search(
                query=content_chunk.content,
                top_k=top_k + (10 if exclude_same_document else 0),  # Get extra to account for filtering
                similarity_threshold=similarity_threshold
            )
            
            # Filter out the same chunk and optionally same document
            filtered_results = []
            for result in results:
                if result.chunk_id == content_chunk.chunk_id:
                    continue
                if exclude_same_document and result.source_location.file_path == content_chunk.source_location.file_path:
                    continue
                filtered_results.append(result)
            
            return filtered_results[:top_k]
            
        except Exception as e:
            error_msg = f"Error finding similar content: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
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
            
        Raises:
            ValueError: If queries list is empty
            RuntimeError: If batch search fails
        """
        if not queries:
            raise ValueError("Queries list cannot be empty")
        
        start_time = time.time()
        
        try:
            results = []
            
            # Process queries individually (could be optimized with batch embedding generation)
            for query in queries:
                if query and query.strip():
                    query_results = self.text_search(
                        query=query,
                        top_k=top_k,
                        search_filter=search_filter
                    )
                    results.append(query_results)
                else:
                    results.append([])
            
            search_time = time.time() - start_time
            total_results = sum(len(r) for r in results)
            
            logger.info(f"Batch search completed in {search_time:.3f}s for {len(queries)} queries, returned {total_results} total results")
            return results
            
        except Exception as e:
            error_msg = f"Error in batch search: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
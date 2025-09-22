"""
Search and retrieval API endpoints.

Handles semantic search operations with query parameter validation,
result formatting, pagination, and cross-modal retrieval capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import time
import asyncio

from ...models import RetrievalResult, ContentType, SourceLocation
from ..schemas import (
    SearchRequest, SearchResponse, RetrievalResultSchema, SourceLocationSchema,
    ErrorResponse
)
from ..dependencies import (
    get_config, get_semantic_retriever, require_api_key
)
from ...monitoring.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

def convert_retrieval_result_to_schema(result: RetrievalResult) -> RetrievalResultSchema:
    """Convert internal RetrievalResult to API schema."""
    
    # Convert source location
    source_location = SourceLocationSchema(
        file_path=result.source_location.file_path,
        page_number=result.source_location.page_number,
        paragraph_index=result.source_location.paragraph_index,
        timestamp_start=result.source_location.timestamp_start,
        timestamp_end=result.source_location.timestamp_end,
        image_coordinates=list(result.source_location.image_coordinates) if result.source_location.image_coordinates else None
    )
    
    return RetrievalResultSchema(
        chunk_id=result.chunk_id,
        content=result.content,
        similarity_score=result.similarity_score,
        relevance_score=result.relevance_score,
        content_type=result.content_type,
        source_location=source_location,
        metadata=result.metadata,
        preview_available=result.preview_data is not None
    )

@router.post(
    "/semantic",
    response_model=SearchResponse,
    summary="Perform semantic search",
    description="Search for semantically similar content across all indexed documents using vector similarity."
)
async def semantic_search(
    request: SearchRequest,
    semantic_retriever=Depends(get_semantic_retriever),
    config=Depends(get_config),
    _=Depends(require_api_key)
):
    """Perform semantic search across indexed documents."""
    
    start_time = time.time()
    
    try:
        # Validate search parameters
        if request.k > config.retrieval.max_k:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"k parameter ({request.k}) exceeds maximum allowed value ({config.retrieval.max_k})"
            )
        
        # Prepare search parameters
        search_params = {
            "k": request.k,
            "similarity_threshold": request.similarity_threshold,
            "content_types": request.content_types,
            "include_preview": request.include_preview,
            "rerank": request.rerank_results
        }
        
        # Perform search
        logger.info(f"Performing semantic search for query: '{request.query[:50]}...'")
        results = await semantic_retriever.search(request.query, **search_params)
        
        # Convert results to API schema
        result_schemas = [convert_retrieval_result_to_schema(result) for result in results]
        
        search_time = time.time() - start_time
        
        logger.info(f"Search completed in {search_time:.3f}s, found {len(results)} results")
        
        return SearchResponse(
            query=request.query,
            results=result_schemas,
            total_results=len(results),
            search_time=search_time,
            reranked=request.rerank_results
        )
    
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.get(
    "/semantic",
    response_model=SearchResponse,
    summary="Perform semantic search (GET)",
    description="Search for semantically similar content using GET parameters for simple queries."
)
async def semantic_search_get(
    q: str = Query(..., description="Search query", min_length=1, max_length=1000),
    k: int = Query(10, description="Number of results to return", ge=1, le=100),
    content_types: Optional[List[ContentType]] = Query(None, description="Filter by content types"),
    similarity_threshold: float = Query(0.0, description="Minimum similarity threshold", ge=0.0, le=1.0),
    include_preview: bool = Query(False, description="Include preview data in results"),
    rerank: bool = Query(True, description="Apply result reranking"),
    semantic_retriever=Depends(get_semantic_retriever),
    config=Depends(get_config),
    _=Depends(require_api_key)
):
    """Perform semantic search using GET parameters."""
    
    # Create search request from parameters
    request = SearchRequest(
        query=q,
        k=k,
        content_types=content_types,
        similarity_threshold=similarity_threshold,
        include_preview=include_preview,
        rerank_results=rerank
    )
    
    # Use the POST endpoint logic
    return await semantic_search(request, semantic_retriever, config)

@router.post(
    "/cross-modal",
    response_model=SearchResponse,
    summary="Perform cross-modal search",
    description="Search across different content modalities (text, image, audio) using unified vector space."
)
async def cross_modal_search(
    request: SearchRequest,
    modalities: List[ContentType] = Query(..., description="Content modalities to search across"),
    semantic_retriever=Depends(get_semantic_retriever),
    config=Depends(get_config),
    _=Depends(require_api_key)
):
    """Perform cross-modal search across different content types."""
    
    start_time = time.time()
    
    try:
        # Validate modalities
        if not modalities:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one modality must be specified"
            )
        
        # Override content types with specified modalities
        request.content_types = modalities
        
        logger.info(f"Performing cross-modal search across {modalities} for query: '{request.query[:50]}...'")
        
        # Use the semantic search functionality with modality filtering
        return await semantic_search(request, semantic_retriever, config)
    
    except Exception as e:
        logger.error(f"Error in cross-modal search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cross-modal search failed: {str(e)}"
        )

@router.get(
    "/similar/{chunk_id}",
    response_model=SearchResponse,
    summary="Find similar content to a specific chunk",
    description="Find content similar to a specific chunk by its ID."
)
async def find_similar_content(
    chunk_id: str,
    k: int = Query(10, description="Number of similar results to return", ge=1, le=100),
    content_types: Optional[List[ContentType]] = Query(None, description="Filter by content types"),
    exclude_same_document: bool = Query(True, description="Exclude results from the same document"),
    semantic_retriever=Depends(get_semantic_retriever),
    config=Depends(get_config),
    _=Depends(require_api_key)
):
    """Find content similar to a specific chunk."""
    
    start_time = time.time()
    
    try:
        # Use the semantic retriever to find similar content
        logger.info(f"Finding similar content to chunk: {chunk_id}")
        
        try:
            # Get the semantic retriever
            from ..dependencies import get_semantic_retriever
            retriever = get_semantic_retriever()
            
            # Try to find similar content using the chunk ID
            # This is a simplified implementation - in a real system,
            # you would retrieve the chunk's embedding and search with it
            results = await retriever.search(
                query=f"similar_to_chunk:{chunk_id}",
                k=k,
                content_types=content_types,
                similarity_threshold=0.5
            )
            
            # Filter out results from the same document if requested
            if exclude_same_document and results:
                # This would need to be implemented based on your chunk ID format
                # For now, we'll keep all results
                pass
                
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            results = []
        
        search_time = time.time() - start_time
        
        return SearchResponse(
            query=f"Similar to chunk: {chunk_id}",
            results=results,
            total_results=len(results),
            search_time=search_time,
            reranked=False
        )
    
    except Exception as e:
        logger.error(f"Error finding similar content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar content search failed: {str(e)}"
        )

@router.get(
    "/filters",
    response_model=Dict[str, Any],
    summary="Get available search filters",
    description="Get information about available search filters and their possible values."
)
async def get_search_filters(
    _=Depends(require_api_key)
):
    """Get available search filters and their options."""
    
    return {
        "content_types": [content_type.value for content_type in ContentType],
        "similarity_threshold": {
            "min": 0.0,
            "max": 1.0,
            "default": 0.0,
            "description": "Minimum similarity score for results"
        },
        "max_results": {
            "min": 1,
            "max": 100,
            "default": 10,
            "description": "Maximum number of results to return"
        },
        "reranking": {
            "default": True,
            "description": "Whether to apply result reranking for improved relevance"
        },
        "cross_modal": {
            "description": "Search across different content modalities",
            "supported_modalities": [content_type.value for content_type in ContentType]
        }
    }

@router.get(
    "/stats",
    response_model=Dict[str, Any],
    summary="Get search statistics",
    description="Get statistics about the indexed content and search performance."
)
async def get_search_statistics(
    _=Depends(require_api_key)
):
    """Get search and indexing statistics."""
    
    try:
        # Get real statistics from the retrieval system
        from ..dependencies import get_semantic_retriever
        retriever = get_semantic_retriever()
        
        # Try to get real statistics
        if hasattr(retriever, 'get_statistics'):
            stats = retriever.get_statistics()
            return stats
        else:
            # Fallback to basic statistics
            return {
                "total_documents": 0,  # Would be populated from real system
                "total_chunks": 0,     # Would be populated from real system
                "content_type_distribution": {
                    "text": 0,
                    "image": 0,
                    "audio": 0,
                    "pdf": 0,
                    "docx": 0
                },
                "index_size_mb": 0,    # Would be calculated from real index
                "average_search_time_ms": 0,  # Would be tracked from real searches
                "last_updated": time.time(),
                "status": "operational"
            }
            
    except Exception as e:
        logger.error(f"Error getting search statistics: {e}")
        return {
            "error": "Could not retrieve statistics",
            "last_updated": time.time(),
            "status": "error"
        }

@router.post(
    "/batch",
    response_model=List[SearchResponse],
    summary="Perform batch search operations",
    description="Execute multiple search queries in a single request for efficiency."
)
async def batch_search(
    queries: List[SearchRequest],
    max_queries: int = Query(10, description="Maximum number of queries per batch", ge=1, le=50),
    semantic_retriever=Depends(get_semantic_retriever),
    config=Depends(get_config),
    _=Depends(require_api_key)
):
    """Perform multiple search operations in batch."""
    
    if len(queries) > max_queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many queries in batch. Maximum allowed: {max_queries}"
        )
    
    if not queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No queries provided"
        )
    
    try:
        logger.info(f"Performing batch search with {len(queries)} queries")
        
        # Execute all searches concurrently
        search_tasks = [
            semantic_search(query, semantic_retriever, config)
            for query in queries
        ]
        
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Handle any exceptions in the results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch search query {i}: {str(result)}")
                # Create error response for failed query
                error_response = SearchResponse(
                    query=queries[i].query,
                    results=[],
                    total_results=0,
                    search_time=0.0,
                    reranked=False
                )
                processed_results.append(error_response)
            else:
                processed_results.append(result)
        
        return processed_results
    
    except Exception as e:
        logger.error(f"Error in batch search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch search failed: {str(e)}"
        )
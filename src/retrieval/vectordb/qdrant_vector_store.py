"""
Qdrant-based vector storage for multimodal content.

Provides fast similarity search for text and image embeddings with local storage
using Qdrant vector database. Supports unified embedding space for cross-modal retrieval.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from typing import List, Dict, Any, Optional, Tuple
import uuid
from pathlib import Path
import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    raise ImportError(
        "Qdrant client not installed. Install with: pip install qdrant-client"
    )

from ...models import ContentChunk, ContentType, RetrievalResult, SourceLocation
from ...config import StorageConfig, EmbeddingConfig
from ...embeddings.unified_embedding_generator import UnifiedEmbeddingGenerator


logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant-based vector storage for multimodal content.
    
    Features:
    - Local vector database storage
    - Fast similarity search
    - Support for text and image embeddings
    - Unified embedding space
    - Metadata filtering
    - Batch operations
    """
    
    def __init__(self, storage_config: StorageConfig, embedding_config: EmbeddingConfig):
        """
        Initialize Qdrant vector store.
        
        Args:
            storage_config: Storage configuration
            embedding_config: Embedding configuration
        """
        self.storage_config = storage_config
        self.embedding_config = embedding_config
        
        # Initialize Qdrant client with local storage
        self.db_path = Path(storage_config.storage_directory) / "qdrant_db"
        self.client = QdrantClient(path=str(self.db_path))
        
        # Collection name for unified embeddings
        self.collection_name = "multimodal_knowledge_base"
        
        # Initialize embedding generator
        self.embedding_generator = UnifiedEmbeddingGenerator(embedding_config)
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
        
        logger.info(f"QdrantVectorStore initialized with database at: {self.db_path}")
    
    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name for col in collections.collections
            )
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_config.embedding_dimension,
                        distance=models.Distance.COSINE
                    )
                )
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def add_chunks(self, chunks: List[ContentChunk]) -> None:
        """
        Add content chunks to the vector store.
        
        Args:
            chunks: List of content chunks with embeddings
        """
        if not chunks:
            return
        
        try:
            # Load embedding model if not loaded
            if not self.embedding_generator.is_loaded:
                self.embedding_generator.load_model()
            
            # Generate embeddings for chunks that don't have them
            chunks_to_embed = [chunk for chunk in chunks if chunk.embedding is None]
            if chunks_to_embed:
                logger.info(f"Generating embeddings for {len(chunks_to_embed)} chunks")
                self.embedding_generator.embed_chunks(chunks_to_embed)
            
            # Prepare points for insertion
            points = []
            for chunk in chunks:
                if chunk.embedding is not None:
                    point = models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=chunk.embedding.tolist(),
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "content": chunk.content,
                            "content_type": chunk.content_type.value,
                            "file_path": chunk.source_location.file_path,
                            "page_number": chunk.source_location.page_number,
                            "paragraph_index": chunk.source_location.paragraph_index,
                            "timestamp_start": chunk.source_location.timestamp_start,
                            "timestamp_end": chunk.source_location.timestamp_end,
                            "confidence_score": chunk.confidence_score,
                            "metadata": chunk.metadata
                        }
                    )
                    points.append(point)
            
            # Insert points in batch
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Successfully added {len(points)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        content_types: Optional[List[ContentType]] = None,
        similarity_threshold: float = 0.5,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Search for relevant content using text query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            content_types: Filter by content types (None for all)
            similarity_threshold: Minimum similarity score
            filter_conditions: Additional filter conditions
            
        Returns:
            List of retrieval results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_text(query)
            
            # Build filter conditions
            query_filter = None
            if content_types or filter_conditions:
                must_conditions = []
                
                if content_types:
                    content_type_values = [ct.value for ct in content_types]
                    must_conditions.append(
                        models.FieldCondition(
                            key="content_type",
                            match=models.MatchValue(value=content_type_values[0])
                            if len(content_type_values) == 1
                            else models.MatchAny(any=content_type_values)
                        )
                    )
                
                if filter_conditions:
                    for key, value in filter_conditions.items():
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
                
                if must_conditions:
                    query_filter = models.Filter(must=must_conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=top_k,
                score_threshold=similarity_threshold
            )
            
            # Convert to RetrievalResult objects
            results = []
            for hit in search_results:
                payload = hit.payload
                
                # Create source location
                source_location = SourceLocation(
                    file_path=payload.get("file_path", ""),
                    page_number=payload.get("page_number"),
                    paragraph_index=payload.get("paragraph_index"),
                    timestamp_start=payload.get("timestamp_start"),
                    timestamp_end=payload.get("timestamp_end")
                )
                
                # Create retrieval result
                result = RetrievalResult(
                    chunk_id=payload.get("chunk_id", ""),
                    content=payload.get("content", ""),
                    similarity_score=float(hit.score),
                    source_location=source_location,
                    content_type=ContentType(payload.get("content_type", "text")),
                    metadata=payload.get("metadata", {}),
                    relevance_score=float(hit.score)  # Use similarity as relevance
                )
                results.append(result)
            
            logger.info(f"Search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def search_by_image(
        self, 
        image_path: str, 
        top_k: int = 10,
        content_types: Optional[List[ContentType]] = None,
        similarity_threshold: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Search for relevant content using image similarity.
        
        Args:
            image_path: Path to query image
            top_k: Number of results to return
            content_types: Filter by content types
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieval results
        """
        try:
            # Generate image embedding
            image_embedding = self.embedding_generator.encode_image(image_path)
            
            # Build filter conditions
            query_filter = None
            if content_types:
                content_type_values = [ct.value for ct in content_types]
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content_type",
                            match=models.MatchValue(value=content_type_values[0])
                            if len(content_type_values) == 1
                            else models.MatchAny(any=content_type_values)
                        )
                    ]
                )
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=image_embedding.tolist(),
                query_filter=query_filter,
                limit=top_k,
                score_threshold=similarity_threshold
            )
            
            # Convert to RetrievalResult objects
            results = []
            for hit in search_results:
                payload = hit.payload
                
                source_location = SourceLocation(
                    file_path=payload.get("file_path", ""),
                    page_number=payload.get("page_number"),
                    paragraph_index=payload.get("paragraph_index"),
                    timestamp_start=payload.get("timestamp_start"),
                    timestamp_end=payload.get("timestamp_end")
                )
                
                result = RetrievalResult(
                    chunk_id=payload.get("chunk_id", ""),
                    content=payload.get("content", ""),
                    similarity_score=float(hit.score),
                    source_location=source_location,
                    content_type=ContentType(payload.get("content_type", "text")),
                    metadata=payload.get("metadata", {}),
                    relevance_score=float(hit.score)
                )
                results.append(result)
            
            logger.info(f"Image search returned {len(results)} results for: {image_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error in image search: {e}")
            raise
    
    def search_cross_modal(
        self, 
        query: str, 
        top_k: int = 10,
        include_images: bool = True,
        similarity_threshold: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Cross-modal search combining text and image results.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            include_images: Whether to include image results
            similarity_threshold: Minimum similarity score
            
        Returns:
            Combined list of retrieval results
        """
        try:
            # Get text-based results
            text_results = self.search(
                query=query,
                top_k=top_k // 2 if include_images else top_k,
                content_types=[ContentType.TEXT, ContentType.AUDIO],
                similarity_threshold=similarity_threshold
            )
            
            results = text_results.copy()
            
            # Add image results if requested
            if include_images:
                image_results = self.search(
                    query=query,
                    top_k=top_k // 2,
                    content_types=[ContentType.IMAGE],
                    similarity_threshold=similarity_threshold
                )
                results.extend(image_results)
            
            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Limit to top_k
            results = results[:top_k]
            
            logger.info(f"Cross-modal search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-modal search: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "status": collection_info.status,
                "database_path": str(self.db_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_document(self, document_id: str) -> None:
        """
        Delete all chunks belonging to a document.
        
        Args:
            document_id: ID of the document to delete
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted document {document_id} from vector store")
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise
    
    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[])
            )
            logger.info("Cleared vector store collection")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            collection_info = self.get_collection_info()
            
            # Get sample points to analyze content types
            sample_results = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000
            )[0]
            
            content_type_counts = {}
            for point in sample_results:
                content_type = point.payload.get("content_type", "unknown")
                content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
            
            return {
                **collection_info,
                "content_type_distribution": content_type_counts,
                "embedding_generator_stats": self.embedding_generator.get_embedding_stats()
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}



# New function to load all embeddings from the local 'embeddings' folder and store them in Qdrant
def load_and_store_local_embeddings():
    """
    Loads all .npy embedding files from the local 'cache/embeddings' folder and stores them in Qdrant.
    Each file should contain a numpy array of shape (n_chunks, embedding_dim) or (embedding_dim,).
    Metadata for each embedding can be loaded from a corresponding .json file if available.
    """
    from src.config import SystemConfig
    import json

    config = SystemConfig()
    vector_store = QdrantVectorStore(
        storage_config=config.storage,
        embedding_config=config.embedding
    )

    embeddings_dir = Path("cache/embeddings")
    npy_files = list(embeddings_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} embedding files in {embeddings_dir}")

    for npy_file in npy_files:
        embeddings = np.load(npy_file)
        # Try to load metadata from a .json file with the same name
        meta_file = npy_file.with_suffix('.json')
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                metas = json.load(f)
        else:
            metas = [{} for _ in range(len(embeddings) if embeddings.ndim == 2 else 1)]

        chunks = []
        if embeddings.ndim == 1:
            # Single embedding
            chunk = ContentChunk(
                chunk_id=str(npy_file.stem),
                document_id=str(npy_file.stem),
                content=f"Embedding from {npy_file.name}",
                content_type=ContentType.TEXT,
                source_location=SourceLocation(file_path=str(npy_file)),
                embedding=embeddings,
                metadata=metas[0] if metas else {}
            )
            chunks.append(chunk)
        else:
            # Multiple embeddings
            for i, emb in enumerate(embeddings):
                chunk = ContentChunk(
                    chunk_id=f"{npy_file.stem}_{i}",
                    document_id=str(npy_file.stem),
                    content=f"Embedding {i} from {npy_file.name}",
                    content_type=ContentType.TEXT,
                    source_location=SourceLocation(file_path=str(npy_file)),
                    embedding=emb,
                    metadata=metas[i] if i < len(metas) else {}
                )
                chunks.append(chunk)

        vector_store.add_chunks(chunks)
        print(f"Stored {len(chunks)} embeddings from {npy_file.name}")

    print("All local embeddings loaded and stored in Qdrant.")


if __name__ == "__main__":
    load_and_store_local_embeddings()

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import json

try:
    from qdrant_client import QdrantClient
    from qdrant_client import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from ...models import ContentChunk, ContentType, RetrievalResult, SourceLocation
from ...config import StorageConfig

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant-based vector storage for pre-generated embeddings.
    
    This class ONLY stores pre-generated embeddings from external sources.
    It does NOT create embeddings - all embeddings must be provided.
    """

    def __init__(self, storage_config: StorageConfig, embedding_dimension: int = 768):
        """
        Initialize Qdrant vector store for storing pre-generated embeddings.
        
        Args:
            storage_config: Storage configuration
            embedding_dimension: Dimension of the embeddings (default: 768)
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required. Install with: pip install qdrant-client")
            
        self.storage_config = storage_config
        self.embedding_dimension = embedding_dimension

        # The location of the stored vector DB
        self.db_path = Path(storage_config.storage_directory) / "qdrant_db"
        self.client: Optional[QdrantClient] = None

        self.collection_name = getattr(storage_config, 'collection_name', 'pre_generated_embeddings')
        
        self.is_initialized = False
        logger.info(f"QdrantVectorStore configured for pre-generated embeddings.")
        logger.info(f"Database location: {self.db_path}")
        logger.info(f"Collection name: {self.collection_name}")
        logger.info(f"Embedding dimension: {self.embedding_dimension}")

    def initialize(self) -> None:
        """Initialize the Qdrant client and collection."""
        if self.is_initialized:
            return
        try:
            if self.client is None:
                logger.info(f"Initializing Qdrant client at path: {self.db_path}")
                # This line sets up the local database
                self.client = QdrantClient(path=str(self.db_path))

            self._ensure_collection_exists()
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise RuntimeError(f"Failed to initialize Qdrant at {self.db_path}")

    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dimension,
                        distance=models.Distance.COSINE
                    )
                )
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error checking or creating collection: {e}")
            raise

    def add_chunks(self, chunks: List[ContentChunk]) -> None:
        """
        Add content chunks with PRE-GENERATED embeddings to the vector store.
        
        This method will NOT generate embeddings. It will raise a ValueError
        if any chunk is missing an embedding.
        
        Args:
            chunks: List of content chunks, each must have a non-None `embedding`.

        Raises:
            ValueError: If any chunk in the list is missing an embedding.
            RuntimeError: If the vector store is not initialized.
        """
        if not self.is_initialized:
            self.initialize()
            
        if not chunks:
            return

        points = []
        for i, chunk in enumerate(chunks):
            # REQUIREMENT: Enforce that embedding already exists.
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk at index {i} (ID: {chunk.chunk_id}) is missing a pre-generated embedding. "
                    "This vector store is configured to only accept chunks with existing embeddings."
                )

            # Ensure embedding is a list of floats for Qdrant
            if isinstance(chunk.embedding, np.ndarray):
                vector = chunk.embedding.tolist()
            else:
                vector = chunk.embedding

            # Create the payload (metadata)
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "content_type": chunk.content_type.value,
                "file_path": getattr(chunk.source_location, 'file_path', None),
                "page_number": getattr(chunk.source_location, 'page_number', None),
                "confidence_score": chunk.confidence_score,
                **chunk.metadata
            }
            
            # Create a Qdrant point
            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload
                )
            )

        # Insert points into the Qdrant collection in a single batch
        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True # Ensures the operation is completed before returning
                )
                logger.info(f"Successfully added {len(points)} chunks with pre-generated embeddings.")
            except Exception as e:
                logger.error(f"Error adding chunks to Qdrant: {e}")
                raise

    def search_by_vector(
        self, 
        query_embedding: Union[List[float], np.ndarray], 
        top_k: int = 10,
        content_types: Optional[List[ContentType]] = None,
        similarity_threshold: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Search for relevant content using a pre-generated query embedding.
        
        Args:
            query_embedding: Pre-generated query embedding vector
            top_k: Number of results to return
            content_types: Filter by content types (None for all)
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieval results
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Convert embedding to list if it's numpy array
            if isinstance(query_embedding, np.ndarray):
                query_vector = query_embedding.tolist()
            else:
                query_vector = query_embedding
            
            # Build filter conditions
            filter_conditions = None
            if content_types:
                filter_conditions = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content_type",
                            match=models.MatchAny(any=[ct.value for ct in content_types])
                        )
                    ]
                )
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_conditions,
                score_threshold=similarity_threshold
            )
            
            # Convert to RetrievalResult objects
            results = []
            for result in search_results:
                payload = result.payload
                source_location = SourceLocation(
                    file_path=payload.get("file_path", ""),
                    page_number=payload.get("page_number")
                )
                
                retrieval_result = RetrievalResult(
                    chunk_id=payload["chunk_id"],
                    content=payload["content"],
                    similarity_score=result.score,
                    source_location=source_location,
                    content_type=ContentType(payload["content_type"]),
                    metadata=payload,
                    relevance_score=result.score
                )
                results.append(retrieval_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self.is_initialized:
            self.initialize()
            
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "collection_name": self.collection_name,
                "is_initialized": self.is_initialized
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        if not self.is_initialized:
            self.initialize()
            
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
            logger.info(f"Deleted all chunks for document: {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            raise

    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        if not self.is_initialized:
            self.initialize()
            
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            logger.info("Cleared all data from collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def get_database_location(self) -> str:
        """
        Get the file system path where the Qdrant vector database is stored.
        
        Returns:
            String path to the local Qdrant database directory
        """
        return str(self.db_path)

    def load_embeddings_from_file(self, embeddings_file_path: str) -> None:
        """
        Load pre-generated embeddings from a file and store them in the vector database.
        
        Expected file format (JSON):
        [
            {
                "chunk_id": "unique_id",
                "document_id": "doc_id", 
                "content": "text content",
                "content_type": "text",
                "embedding": [0.1, 0.2, ...],
                "metadata": {...},
                "source_location": {"file_path": "...", "page_number": 1},
                "confidence_score": 0.95
            },
            ...
        ]
        
        Args:
            embeddings_file_path: Path to the JSON file containing embeddings
        """
        try:
            with open(embeddings_file_path, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            chunks = []
            for item in embeddings_data:
                # Create ContentChunk objects from the loaded data
                source_location = SourceLocation(
                    file_path=item.get('source_location', {}).get('file_path', ''),
                    page_number=item.get('source_location', {}).get('page_number')
                )
                
                chunk = ContentChunk(
                    chunk_id=item['chunk_id'],
                    document_id=item['document_id'],
                    content=item['content'],
                    content_type=ContentType(item['content_type']),
                    embedding=item['embedding'],  # Pre-generated embedding
                    source_location=source_location,
                    metadata=item.get('metadata', {}),
                    confidence_score=item.get('confidence_score', 1.0)
                )
                chunks.append(chunk)
            
            # Store the chunks with embeddings
            self.add_chunks(chunks)
            logger.info(f"Successfully loaded {len(chunks)} embeddings from {embeddings_file_path}")
            
        except Exception as e:
            logger.error(f"Error loading embeddings from file {embeddings_file_path}: {e}")
            raise

    def load_embeddings_from_numpy(self, embeddings_array: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        """
        Load pre-generated embeddings from numpy array with corresponding metadata.
        
        Args:
            embeddings_array: Numpy array of shape (n_samples, embedding_dim)
            metadata_list: List of metadata dictionaries, one for each embedding
        """
        if len(embeddings_array) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        chunks = []
        for i, (embedding, metadata) in enumerate(zip(embeddings_array, metadata_list)):
            source_location = SourceLocation(
                file_path=metadata.get('file_path', ''),
                page_number=metadata.get('page_number')
            )
            
            chunk = ContentChunk(
                chunk_id=metadata.get('chunk_id', f'chunk_{i}'),
                document_id=metadata.get('document_id', f'doc_{i}'),
                content=metadata.get('content', ''),
                content_type=ContentType(metadata.get('content_type', 'text')),
                embedding=embedding,  # Pre-generated embedding
                source_location=source_location,
                metadata=metadata.get('additional_metadata', {}),
                confidence_score=metadata.get('confidence_score', 1.0)
            )
            chunks.append(chunk)
        
        # Store the chunks with embeddings
        self.add_chunks(chunks)
        logger.info(f"Successfully loaded {len(chunks)} embeddings from numpy array")

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the vector database.
        
        Returns:
            Dictionary containing database location, statistics, and configuration
        """
        info = {
            "database_location": self.get_database_location(),
            "collection_name": self.collection_name,
            "embedding_dimension": self.embedding_dimension,
            "is_initialized": self.is_initialized
        }
        
        # Add statistics if initialized
        if self.is_initialized:
            info.update(self.get_statistics())
        
        return info
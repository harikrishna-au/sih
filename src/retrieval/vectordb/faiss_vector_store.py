"""
FAISS Vector Store for Pre-Generated Embeddings

This vector store is designed to ONLY accept and store pre-generated embeddings.
It does NOT generate embeddings - all embeddings must be provided externally.
Uses FAISS for efficient similarity search.
"""

import logging
import json
import pickle
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ...models import ContentChunk, ContentType, RetrievalResult, SourceLocation
from ...config import StorageConfig

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector storage for pre-generated embeddings.
    
    This class ONLY stores pre-generated embeddings from external sources.
    It does NOT create embeddings - all embeddings must be provided.
    
    Key Features:
    - Accepts only pre-generated embeddings
    - Stores embeddings using FAISS for fast similarity search
    - Provides database location for retrieval
    - No UUID requirements - uses simple integer indices
    """

    def __init__(self, storage_config: StorageConfig, embedding_dimension: int = 384):
        """
        Initialize FAISS vector store for storing pre-generated embeddings.
        
        Args:
            storage_config: Storage configuration
            embedding_dimension: Dimension of the embeddings (default: 384)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
            
        self.storage_config = storage_config
        self.embedding_dimension = embedding_dimension

        # Collection name for compatibility
        self.collection_name = getattr(storage_config, 'collection_name', 'faiss_embeddings')

        # The location of the stored vector DB
        self.db_path = Path(storage_config.storage_directory) / "faiss_db"
        self.index_file = self.db_path / "index.faiss"
        self.metadata_file = self.db_path / "metadata.pkl"
        
        # FAISS index and metadata storage
        self.index = None
        self.metadata = []  # List to store chunk metadata
        self.chunk_id_to_idx = {}  # Map chunk_id to index position
        
        self.is_initialized = False
        
        logger.info(f"FAISSVectorStore configured for pre-generated embeddings")
        logger.info(f"Database location: {self.db_path}")
        logger.info(f"Embedding dimension: {self.embedding_dimension}")

    def initialize(self) -> None:
        """Initialize the FAISS index."""
        if self.is_initialized:
            return
            
        try:
            # Create database directory if it doesn't exist
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Try to load existing index
            if self.index_file.exists() and self.metadata_file.exists():
                logger.info("Loading existing FAISS index")
                self._load_index()
            else:
                logger.info("Creating new FAISS index")
                self._create_new_index()
            
            self.is_initialized = True
            logger.info("FAISS vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise RuntimeError(f"Failed to initialize FAISS at {self.db_path}: {e}")

    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        # Create a flat L2 index (exact search)
        # For larger datasets, you might want to use IndexIVFFlat or IndexHNSWFlat
        self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product (cosine similarity)
        self.metadata = []
        self.chunk_id_to_idx = {}
        logger.info(f"Created new FAISS index with dimension {self.embedding_dimension}")

    def _load_index(self) -> None:
        """Load existing FAISS index and metadata."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            with open(self.metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.chunk_id_to_idx = data['chunk_id_to_idx']
            
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            # If loading fails, create new index
            self._create_new_index()

    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            data = {
                'metadata': self.metadata,
                'chunk_id_to_idx': self.chunk_id_to_idx
            }
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
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
            logger.warning("No chunks provided to add_chunks")
            return

        logger.info(f"Adding {len(chunks)} chunks with pre-generated embeddings")
        
        vectors = []
        chunk_metadata = []
        
        for i, chunk in enumerate(chunks):
            try:
                # REQUIREMENT: Enforce that embedding already exists
                if chunk.embedding is None:
                    raise ValueError(
                        f"Chunk at index {i} (ID: {chunk.chunk_id}) is missing a pre-generated embedding. "
                        "This vector store is configured to only accept chunks with existing embeddings."
                    )

                # Ensure embedding is a numpy array
                if isinstance(chunk.embedding, list):
                    vector = np.array(chunk.embedding, dtype=np.float32)
                else:
                    vector = chunk.embedding.astype(np.float32)
                
                # Validate embedding dimension
                if len(vector) != self.embedding_dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch for chunk {chunk.chunk_id}. "
                        f"Expected {self.embedding_dimension}, got {len(vector)}"
                    )

                # Normalize vector for cosine similarity (FAISS IndexFlatIP expects normalized vectors)
                vector = vector / np.linalg.norm(vector)
                
                vectors.append(vector)
                
                # Create metadata for this chunk
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "content_type": chunk.content_type.value,
                    "confidence_score": chunk.confidence_score,
                    "file_path": chunk.source_location.file_path if chunk.source_location else None,
                    "page_number": chunk.source_location.page_number if chunk.source_location else None,
                }
                
                # Add chunk metadata
                if chunk.metadata:
                    for key, value in chunk.metadata.items():
                        if key not in metadata:
                            metadata[key] = value
                
                chunk_metadata.append(metadata)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i} (ID: {getattr(chunk, 'chunk_id', 'unknown')}): {e}")
                raise

        # Add vectors to FAISS index
        if vectors:
            try:
                vectors_array = np.vstack(vectors)
                
                # Get starting index for new vectors
                start_idx = self.index.ntotal
                
                # Add vectors to index
                self.index.add(vectors_array)
                
                # Update metadata and chunk_id mapping
                for i, (chunk_metadata_item, chunk) in enumerate(zip(chunk_metadata, chunks)):
                    idx = start_idx + i
                    self.metadata.append(chunk_metadata_item)
                    self.chunk_id_to_idx[chunk.chunk_id] = idx
                
                # Save index to disk
                self._save_index()
                
                logger.info(f"Successfully added {len(vectors)} chunks with pre-generated embeddings")
                
            except Exception as e:
                logger.error(f"Error adding chunks to FAISS: {e}")
                raise RuntimeError(f"Failed to store embeddings in FAISS: {e}")

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
        
        if self.index.ntotal == 0:
            logger.warning("No vectors in index for search")
            return []
        
        try:
            # Convert embedding to numpy array
            if isinstance(query_embedding, list):
                query_vector = np.array(query_embedding, dtype=np.float32)
            else:
                query_vector = query_embedding.astype(np.float32)
            
            # Validate query embedding dimension
            if len(query_vector) != self.embedding_dimension:
                raise ValueError(
                    f"Query embedding dimension mismatch. "
                    f"Expected {self.embedding_dimension}, got {len(query_vector)}"
                )
            
            # Normalize query vector for cosine similarity
            query_vector = query_vector / np.linalg.norm(query_vector)
            
            # Reshape for FAISS (needs 2D array)
            query_vector = query_vector.reshape(1, -1)
            
            # Search in FAISS index
            logger.info(f"Searching with top_k={top_k}, threshold={similarity_threshold}")
            
            # Get more results than needed for filtering
            search_k = min(top_k * 2, self.index.ntotal)
            scores, indices = self.index.search(query_vector, search_k)
            
            # Convert to results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                    
                if score < similarity_threshold:
                    continue
                
                if idx >= len(self.metadata):
                    logger.warning(f"Index {idx} out of range for metadata")
                    continue
                
                metadata = self.metadata[idx]
                
                # Apply content type filter
                if content_types:
                    chunk_content_type = ContentType(metadata["content_type"])
                    if chunk_content_type not in content_types:
                        continue
                
                # Create source location
                source_location = SourceLocation(
                    file_path=metadata.get("file_path", ""),
                    page_number=metadata.get("page_number")
                )
                
                # Create retrieval result
                result = RetrievalResult(
                    chunk_id=metadata["chunk_id"],
                    content=metadata["content"],
                    similarity_score=float(score),
                    source_location=source_location,
                    content_type=ContentType(metadata["content_type"]),
                    metadata=metadata,
                    relevance_score=float(score)
                )
                results.append(result)
                
                # Stop if we have enough results
                if len(results) >= top_k:
                    break
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            raise

    def get_database_location(self) -> str:
        """
        Get the file system path where the FAISS vector database is stored.
        
        Returns:
            String path to the local FAISS database directory
        """
        return str(self.db_path.absolute())

    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self.is_initialized:
            self.initialize()
            
        try:
            return {
                "vectors_count": self.index.ntotal if self.index else 0,
                "vector_dimension": self.embedding_dimension,
                "index_type": type(self.index).__name__ if self.index else "None",
                "is_initialized": self.is_initialized,
                "database_location": self.get_database_location(),
                "metadata_count": len(self.metadata)
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {
                "error": str(e),
                "database_location": self.get_database_location(),
                "is_initialized": self.is_initialized
            }

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the vector database.
        
        Returns:
            Dictionary containing database location, statistics, and configuration
        """
        info = {
            "database_location": self.get_database_location(),
            "embedding_dimension": self.embedding_dimension,
            "is_initialized": self.is_initialized,
            "index_file": str(self.index_file),
            "metadata_file": str(self.metadata_file)
        }
        
        # Add statistics if initialized
        if self.is_initialized:
            stats = self.get_statistics()
            info.update(stats)
        
        return info

    def load_embeddings_from_file(self, embeddings_file_path: str) -> None:
        """
        Load pre-generated embeddings from a JSON file and store them in the vector database.
        
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
            logger.info(f"Loading embeddings from file: {embeddings_file_path}")
            
            with open(embeddings_file_path, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            chunks = []
            for item in embeddings_data:
                # Create source location
                source_location = None
                if 'source_location' in item and item['source_location']:
                    source_location = SourceLocation(
                        file_path=item['source_location'].get('file_path', ''),
                        page_number=item['source_location'].get('page_number')
                    )
                
                # Create content chunk
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

    def load_embeddings_from_numpy(
        self, 
        embeddings_array: np.ndarray, 
        metadata_list: List[Dict[str, Any]]
    ) -> None:
        """
        Load pre-generated embeddings from numpy array with corresponding metadata.
        
        Args:
            embeddings_array: Numpy array of shape (n_samples, embedding_dim)
            metadata_list: List of metadata dictionaries, one for each embedding
        """
        if len(embeddings_array) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        logger.info(f"Loading {len(embeddings_array)} embeddings from numpy array")
        
        chunks = []
        for i, (embedding, metadata) in enumerate(zip(embeddings_array, metadata_list)):
            # Create source location
            source_location = None
            if 'file_path' in metadata:
                source_location = SourceLocation(
                    file_path=metadata['file_path'],
                    page_number=metadata.get('page_number')
                )
            
            # Create content chunk
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

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        if not self.is_initialized:
            self.initialize()
            
        logger.warning("Document deletion not efficiently supported in FAISS. Consider rebuilding index.")
        # FAISS doesn't support efficient deletion, so we'd need to rebuild the entire index
        # For now, just log a warning
        logger.info(f"Document deletion requested for: {document_id} (not implemented)")

    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        if not self.is_initialized:
            self.initialize()
            
        try:
            logger.info("Clearing all data from FAISS index")
            
            # Create new empty index
            self._create_new_index()
            
            # Save empty index
            self._save_index()
            
            logger.info("Successfully cleared all data from FAISS index")
            
        except Exception as e:
            logger.error(f"Error clearing FAISS index: {e}")
            raise

    def close(self) -> None:
        """Close and cleanup the FAISS index."""
        if self.index:
            try:
                # Save index before closing
                self._save_index()
                logger.info("FAISS index saved and closed")
            except Exception as e:
                logger.warning(f"Error saving FAISS index on close: {e}")
            finally:
                self.index = None
                self.metadata = []
                self.chunk_id_to_idx = {}
                self.is_initialized = False
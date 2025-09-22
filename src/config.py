"""
Configuration management system for the multimodal RAG backend.

Provides dataclasses for all system configurations including processing,
embedding generation, LLM settings, and storage options.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
from pathlib import Path


@dataclass
class ProcessingConfig:
    """Configuration for document processing operations."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size_mb: int = 100
    supported_formats: List[str] = field(default_factory=lambda: [
        'pdf', 'docx', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'm4a'
    ])
    batch_size: int = 10
    max_concurrent_files: int = 4
    temp_directory: str = "/tmp/multimodal_rag"
    preserve_formatting: bool = True
    extract_images_from_pdf: bool = True
    ocr_confidence_threshold: float = 0.7


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model_name: str = "ViT-B/32"
    embedding_dimension: int = 384
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    embedding_cache_dir: str = "cache/embeddings"
    device: str = "auto"  # auto, cpu, cuda
    max_sequence_length: int = 512


@dataclass
class LLMConfig:
    """Configuration for local LLM integration."""
    model_path: str = "models/llama-2-7b-chat.Q4_K_M.gguf"
    max_context_length: int = 4096
    max_response_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    quantization: str = "4bit"
    device: str = "auto"
    n_threads: int = -1  # -1 for auto
    context_window_strategy: str = "sliding"  # sliding, truncate
    citation_prompt_template: str = """Based on the following context, answer the question and include numbered citations [1], [2], etc. for each source used.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class StorageConfig:
    """Configuration for vector storage and indexing."""
    index_type: str = "IVFFlat"  # IVFFlat, HNSW, Flat
    index_params: Dict[str, Any] = field(default_factory=lambda: {
        "nlist": 100,  # For IVF indices
        "M": 16,       # For HNSW indices
        "efConstruction": 200,  # For HNSW indices
    })
    storage_directory: str = "storage"
    metadata_db_path: str = "storage/metadata.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    compression_enabled: bool = True
    index_rebuild_threshold: float = 0.1  # Rebuild if 10% of vectors are deleted


@dataclass
class RetrievalConfig:
    """Configuration for semantic retrieval operations."""
    default_k: int = 10
    max_k: int = 100
    similarity_threshold: float = 0.5
    reranking_enabled: bool = True
    cross_modal_search: bool = True
    result_diversification: bool = True
    diversification_lambda: float = 0.5
    search_timeout_seconds: float = 30.0
    enable_query_expansion: bool = False


@dataclass
class APIConfig:
    """Configuration for the REST API layer."""
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size_mb: int = 100
    request_timeout_seconds: int = 300
    rate_limiting_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    api_key_required: bool = False
    api_key: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for system logging."""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_file: Optional[str] = "logs/multimodal_rag.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    log_performance_metrics: bool = True


@dataclass
class SystemConfig:
    """Main system configuration containing all subsystem configs."""
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    data_directory: str = "data"
    models_directory: str = "models"
    cache_directory: str = "cache"
    offline_mode: bool = True
    validate_models_on_startup: bool = True
    auto_create_directories: bool = True


class ConfigManager:
    """
    Manages system configuration loading, validation, and updates.
    
    Supports loading from environment variables, configuration files,
    and runtime updates with validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self._config: Optional[SystemConfig] = None
    
    def load_config(self) -> SystemConfig:
        """Load configuration from file and environment variables."""
        if self._config is None:
            self._config = SystemConfig()
            self._apply_environment_overrides()
            self._validate_config()
            self._create_directories()
        return self._config
    
    def _apply_environment_overrides(self) -> None:
        """Apply configuration overrides from environment variables."""
        if not self._config:
            return
            
        # Processing config overrides
        if os.getenv("CHUNK_SIZE"):
            self._config.processing.chunk_size = int(os.getenv("CHUNK_SIZE"))
        if os.getenv("MAX_FILE_SIZE_MB"):
            self._config.processing.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB"))
        
        # Embedding config overrides
        if os.getenv("TEXT_MODEL_NAME"):
            self._config.embedding.text_model_name = os.getenv("TEXT_MODEL_NAME")
        if os.getenv("EMBEDDING_DEVICE"):
            self._config.embedding.device = os.getenv("EMBEDDING_DEVICE")
        
        # LLM config overrides
        if os.getenv("LLM_MODEL_PATH"):
            self._config.llm.model_path = os.getenv("LLM_MODEL_PATH")
        if os.getenv("LLM_TEMPERATURE"):
            self._config.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
        
        # API config overrides
        if os.getenv("API_HOST"):
            self._config.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            self._config.api.port = int(os.getenv("API_PORT"))
    
    def _validate_config(self) -> None:
        """Validate configuration values and constraints."""
        if not self._config:
            return
            
        # Validate processing config
        if self._config.processing.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self._config.processing.chunk_overlap >= self._config.processing.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        # Validate embedding config
        if self._config.embedding.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        
        # Validate LLM config
        if self._config.llm.temperature < 0 or self._config.llm.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        
        # Validate storage paths
        if not self._config.storage.storage_directory:
            raise ValueError("storage_directory cannot be empty")
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        if not self._config or not self._config.auto_create_directories:
            return
            
        directories = [
            self._config.data_directory,
            self._config.models_directory,
            self._config.cache_directory,
            self._config.storage.storage_directory,
            self._config.embedding.embedding_cache_dir,
            self._config.processing.temp_directory,
        ]
        
        # Add log directory if file logging is enabled
        if self._config.logging.log_file:
            log_dir = Path(self._config.logging.log_file).parent
            directories.append(str(log_dir))
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values at runtime."""
        if not self._config:
            self.load_config()
        
        # Update nested configuration values
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        self._validate_config()
    
    def get_config(self) -> SystemConfig:
        """Get the current system configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
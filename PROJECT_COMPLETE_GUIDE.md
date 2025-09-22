# 🚀 Multimodal RAG System - Complete Project Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Running the Project](#running-the-project)
5. [Pipeline Workflow](#pipeline-workflow)
6. [API Usage](#api-usage)
7. [Vector Database Operations](#vector-database-operations)
8. [Retrieving Stored Embeddings](#retrieving-stored-embeddings)
9. [Testing & Verification](#testing--verification)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This is a **Multimodal RAG (Retrieval-Augmented Generation) System** that processes documents, generates embeddings, and stores them in a local FAISS vector database for fast similarity search and retrieval.

### Key Features
- ✅ **Document Processing**: PDF, DOCX, TXT file support
- ✅ **Embedding Generation**: Uses sentence-transformers for text embeddings
- ✅ **Local Vector Storage**: FAISS database for fast similarity search
- ✅ **RESTful API**: FastAPI-based endpoints for document upload and search
- ✅ **No UUID Dependencies**: Simple integer indexing
- ✅ **Pre-generated Embeddings**: Accepts only pre-generated embeddings

---

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   PDF Upload    │───▶│  Document        │───▶│   Embedding     │───▶│   FAISS Vector   │
│   (FastAPI)     │    │  Chunking        │    │   Generation    │    │   Database       │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
                                │                        │                        │
                                ▼                        ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
                       │ Text Processor   │    │ Sentence        │    │ index.faiss      │
                       │ PDF Processor    │    │ Transformers    │    │ metadata.pkl     │
                       │ DOCX Processor   │    │ (all-MiniLM)    │    │ (Local Storage)  │
                       └──────────────────┘    └─────────────────┘    └──────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ Search & Query  │
                                               │ (Similarity)    │
                                               └─────────────────┘
```

### Core Components

1. **FastAPI Server** (`src/api/`)
   - Document upload endpoints
   - Search endpoints
   - Status monitoring

2. **Document Processors** (`src/processors/`)
   - PDF processor
   - DOCX processor  
   - Text processor
   - Document routing

3. **Embedding Pipeline** (`src/pipeline/`)
   - Unified processing pipeline
   - Embedding generation
   - Vector storage integration

4. **FAISS Vector Store** (`src/retrieval/vectordb/`)
   - Local vector database
   - Similarity search
   - Metadata management

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- At least 4GB RAM (for embedding models)

### 1. Clone and Setup
```bash
# Navigate to project directory
cd MultiRAG-SIH-2025

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install core dependencies
pip install fastapi uvicorn python-multipart
pip install sentence-transformers faiss-cpu
pip install numpy pandas pathlib
pip install pydantic python-dotenv
```

### 3. Create Required Directories
```bash
# The system will auto-create these, but you can create manually:
mkdir -p data models cache storage logs temp
```

---

## 🚀 Running the Project

### Method 1: Using the Run Server Script (Recommended)
```bash
python run_server.py
```

### Method 2: Direct FastAPI Command
```bash
python -c "from src.api.main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
```

### Method 3: Background Process (Linux/Mac)
```bash
nohup python3 -c "from src.api.main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)" > server.log 2>&1 &
```

### Method 4: Background Process (Windows PowerShell)
```powershell
Start-Process -FilePath "python" -ArgumentList "run_server.py" -WindowStyle Hidden
```

### Server Endpoints
Once running, access:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root Endpoint**: http://localhost:8000/

---

## 🔄 Pipeline Workflow

### Step-by-Step Process

#### 1. Document Upload
```
POST /api/v1/documents/upload
```
- Upload PDF/DOCX/TXT file
- Returns job ID for tracking
- File saved temporarily for processing

#### 2. Document Processing
```
Document → Text Extraction → Chunking → Embedding Generation → Vector Storage
```

**Detailed Steps:**
1. **File Validation**: Check format, size, permissions
2. **Text Extraction**: Extract content from document
3. **Chunking**: Split into smaller segments (configurable size)
4. **Embedding Generation**: Convert text chunks to 384-dimensional vectors
5. **Vector Storage**: Store in FAISS database with metadata

#### 3. Storage Structure
```
storage/faiss_db/
├── index.faiss      # FAISS vector index (binary)
└── metadata.pkl     # Chunk metadata (pickle format)
```

#### 4. Search & Retrieval
```
Query Text → Embedding → Similarity Search → Ranked Results
```

---

## 📡 API Usage

### 1. Upload Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "filename": "your_document.pdf",
  "file_size": 1024000,
  "status": "pending",
  "message": "File uploaded successfully, processing started"
}
```

### 2. Check Processing Status
```bash
curl -X GET "http://localhost:8000/api/v1/documents/status/{job_id}"
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "total_files": 1,
  "processed_files": 1,
  "failed_files": 0,
  "progress_percentage": 100.0,
  "processing_time": 15.5
}
```

### 3. Search Documents
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is machine learning?",
       "top_k": 5,
       "similarity_threshold": 0.5
     }'
```

**Response:**
```json
{
  "query": "What is machine learning?",
  "results": [
    {
      "chunk_id": "doc_123_chunk_001",
      "content": "Machine learning is a subset of artificial intelligence...",
      "similarity_score": 0.85,
      "source_file": "document.pdf",
      "page_number": 1
    }
  ],
  "total_results": 5,
  "search_time": 0.05
}
```

---

## 🗄️ Vector Database Operations

### Database Location
```
storage/faiss_db/
├── index.faiss      # Vector index (binary format)
└── metadata.pkl     # Metadata (Python pickle)
```

### Database Statistics
```python
from src.retrieval.vectordb.faiss_vector_store import FAISSVectorStore
from src.config import ConfigManager

config = ConfigManager().load_config()
vector_store = FAISSVectorStore(config.storage, embedding_dimension=384)

# Get statistics
stats = vector_store.get_statistics()
print(f"Vectors: {stats['vectors_count']}")
print(f"Dimension: {stats['vector_dimension']}")
print(f"Location: {stats['database_location']}")
```

### Manual Vector Operations
```python
# Initialize vector store
vector_store = FAISSVectorStore(config.storage, embedding_dimension=384)

# Add pre-generated embeddings
chunks = [...]  # List of ContentChunk objects with embeddings
vector_store.add_chunks(chunks)

# Search by vector
query_embedding = model.encode(["your query"])[0]
results = vector_store.search_by_vector(query_embedding, top_k=10)

# Load from JSON file
vector_store.load_embeddings_from_file("embeddings.json")
```

---

## 🔍 Retrieving Stored Embeddings

### Method 1: Direct Database Access
```python
import pickle
import faiss
import numpy as np

# Load FAISS index
index = faiss.read_index("storage/faiss_db/index.faiss")

# Load metadata
with open("storage/faiss_db/metadata.pkl", "rb") as f:
    data = pickle.load(f)
    metadata = data['metadata']
    chunk_id_to_idx = data['chunk_id_to_idx']

print(f"Total vectors: {index.ntotal}")
print(f"Vector dimension: {index.d}")
print(f"Metadata entries: {len(metadata)}")

# Get specific vector by index
vector_idx = 0
vector = index.reconstruct(vector_idx)
chunk_metadata = metadata[vector_idx]

print(f"Vector shape: {vector.shape}")
print(f"Chunk ID: {chunk_metadata['chunk_id']}")
print(f"Content: {chunk_metadata['content'][:100]}...")
```

### Method 2: Using Vector Store API
```python
from src.retrieval.vectordb.faiss_vector_store import FAISSVectorStore
from src.config import ConfigManager

# Initialize
config = ConfigManager().load_config()
vector_store = FAISSVectorStore(config.storage, embedding_dimension=384)

# Get all statistics
stats = vector_store.get_statistics()
print(f"Database info: {stats}")

# Search to retrieve specific embeddings
query_embedding = np.random.rand(384)  # Your query vector
results = vector_store.search_by_vector(query_embedding, top_k=10)

for result in results:
    print(f"Chunk ID: {result.chunk_id}")
    print(f"Content: {result.content}")
    print(f"Similarity: {result.similarity_score}")
    print(f"Metadata: {result.metadata}")
    print("-" * 50)
```

### Method 3: Export Embeddings
```python
def export_all_embeddings(output_file="exported_embeddings.json"):
    """Export all embeddings to JSON file"""
    import json
    
    # Load data
    with open("storage/faiss_db/metadata.pkl", "rb") as f:
        data = pickle.load(f)
        metadata = data['metadata']
    
    index = faiss.read_index("storage/faiss_db/index.faiss")
    
    # Export data
    exported_data = []
    for i in range(index.ntotal):
        vector = index.reconstruct(i).tolist()
        chunk_data = metadata[i].copy()
        chunk_data['embedding'] = vector
        exported_data.append(chunk_data)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(exported_data, f, indent=2)
    
    print(f"Exported {len(exported_data)} embeddings to {output_file}")

# Usage
export_all_embeddings()
```

---

## 🧪 Testing & Verification

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Test Document Upload
```python
import requests

# Upload test document
with open("test_document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/documents/upload",
        files={"file": f}
    )
    
job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")

# Check status
status_response = requests.get(f"http://localhost:8000/api/v1/documents/status/{job_id}")
print(f"Status: {status_response.json()}")
```

### 3. Test Search
```python
# Search test
search_response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={
        "query": "machine learning",
        "top_k": 5,
        "similarity_threshold": 0.5
    }
)

results = search_response.json()
print(f"Found {len(results['results'])} results")
```

### 4. Verify Vector Database
```python
from src.retrieval.vectordb.faiss_vector_store import FAISSVectorStore
from src.config import ConfigManager

config = ConfigManager().load_config()
vector_store = FAISSVectorStore(config.storage, embedding_dimension=384)

# Check if database exists and has data
stats = vector_store.get_statistics()
print(f"Vectors stored: {stats.get('vectors_count', 0)}")
print(f"Database location: {stats.get('database_location')}")

if stats.get('vectors_count', 0) > 0:
    print("✅ Vector database is working and contains embeddings!")
else:
    print("❌ Vector database is empty")
```

---

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Processing Configuration
CHUNK_SIZE=512
MAX_FILE_SIZE_MB=100

# Embedding Configuration
TEXT_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_DEVICE=cpu

# Storage Configuration
STORAGE_DIRECTORY=storage
```

### Configuration Files
Main configuration in `src/config.py`:
```python
@dataclass
class ProcessingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size_mb: int = 100
    supported_formats: List[str] = ['pdf', 'docx', 'txt']

@dataclass
class EmbeddingConfig:
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    device: str = "cpu"
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Memory Errors
**Problem**: `MemoryError: bad allocation`
**Solution**: 
- Use CPU instead of GPU: Set `device: "cpu"` in config
- Reduce batch size: Set `batch_size: 16` or lower
- Use smaller embedding model

#### 2. Module Not Found
**Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**:
```bash
# Ensure you're in the project root directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or on Windows:
set PYTHONPATH=%PYTHONPATH%;%CD%
```

#### 3. Port Already in Use
**Problem**: `Address already in use`
**Solution**:
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
# Or use different port
uvicorn.run(app, host="0.0.0.0", port=8001)
```

#### 4. FAISS Installation Issues
**Problem**: FAISS installation fails
**Solution**:
```bash
# For CPU-only version
pip install faiss-cpu

# For GPU version (if CUDA available)
pip install faiss-gpu
```

#### 5. Empty Vector Database
**Problem**: No embeddings stored
**Solution**:
- Check processing logs for errors
- Verify document format is supported
- Ensure embedding model loads correctly
- Check file permissions

### Debug Commands
```bash
# Check server logs
tail -f server.log

# Test embedding generation
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); print('Model loaded successfully')"

# Check vector database
python -c "import faiss; index = faiss.read_index('storage/faiss_db/index.faiss'); print(f'Vectors: {index.ntotal}')"
```

---

## 📊 Performance Metrics

### Typical Performance
- **Document Processing**: 2-5 seconds per page
- **Embedding Generation**: 100-500 chunks per second
- **Vector Search**: <50ms for 10K vectors
- **Storage Efficiency**: ~1.5KB per vector (384-dim)

### Scaling Guidelines
- **Small Dataset**: <1K documents → Single instance
- **Medium Dataset**: 1K-10K documents → Consider batch processing
- **Large Dataset**: >10K documents → Distributed processing

---

## 🎯 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run_server.py"]
```

### Environment Setup
```bash
# Production environment
export ENVIRONMENT=production
export API_DEBUG=false
export LOG_LEVEL=INFO
```

### Monitoring
- Health endpoint: `/health`
- Metrics endpoint: `/metrics` (if implemented)
- Log files: `logs/multimodal_rag.log`

---

## 📝 Summary

This Multimodal RAG System provides:

1. **✅ Complete Document Processing Pipeline**
2. **✅ Local FAISS Vector Database Storage**  
3. **✅ RESTful API for Upload and Search**
4. **✅ Pre-generated Embedding Support**
5. **✅ Fast Similarity Search and Retrieval**

### Quick Start Commands
```bash
# 1. Install dependencies
pip install fastapi uvicorn python-multipart sentence-transformers faiss-cpu

# 2. Start server
python run_server.py

# 3. Upload document
curl -X POST "http://localhost:8000/api/v1/documents/upload" -F "file=@document.pdf"

# 4. Search
curl -X POST "http://localhost:8000/api/v1/search" -H "Content-Type: application/json" -d '{"query": "your search query"}'
```

**🎉 Your vector database is ready and working!** 🚀
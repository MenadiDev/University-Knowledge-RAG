"""Central settings for the RAG app. Change models or paths here only."""

from pathlib import Path

# Project root (the folder that contains app.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# Folders
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
STYLE_FILE = BASE_DIR / "assets" / "style.css"

# Vector store
COLLECTION_NAME = "university_docs"

# Ollama models
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "llama3.2:3b"
LLM_TEMPERATURE = 0

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval: how many chunks to pass to the model
TOP_K = 4

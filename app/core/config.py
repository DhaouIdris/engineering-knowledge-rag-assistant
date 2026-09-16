from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # FAISS
    faiss_index_path: str = "storage/faiss_index"
    documents_path: str = "data/documents"
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    
    # Retriever
    retriever_k: int = 4
    
    # App
    app_name: str = "Multi-Agent RAG Assistant"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
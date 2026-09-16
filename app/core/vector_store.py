import os
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from app.core.embeddings import get_embeddings
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def build_vector_store(chunks: List[Document]) -> FAISS:
    logger.info("Building FAISS index...")
    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    
    path = settings.faiss_index_path
    Path(path).mkdir(parents=True, exist_ok=True)
    store.save_local(path)
    logger.info(f"FAISS index saved to {path}")
    return store

def load_vector_store() -> FAISS:
    path = settings.faiss_index_path
    if not Path(path).exists():
        raise FileNotFoundError(f"No FAISS index at {path}. Run indexing first.")
    
    logger.info(f"Loading FAISS index from {path}")
    return FAISS.load_local(
        path, get_embeddings(), allow_dangerous_deserialization=True
    )
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

def load_documents(folder_path: str = None) -> List[Document]:
    folder = Path(folder_path or settings.documents_path)
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF found in {folder}")
        return []
    
    docs = []
    for pdf in pdf_files:
        logger.info(f"Loading: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    
    logger.info(f"Loaded {len(docs)} pages from {len(pdf_files)} PDFs")
    return docs

def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks
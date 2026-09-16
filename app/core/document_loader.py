from pathlib import Path
from typing import Collection, List
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

def load_documents(
    folder_path: str = None,
    filenames: Collection[str] | None = None,
) -> List[Document]:
    folder = Path(folder_path or settings.documents_path)
    allowed = set(filenames) if filenames is not None else None
    pdf_files = sorted(
        (pdf for pdf in folder.glob("*.pdf") if allowed is None or pdf.name in allowed),
        key=lambda path: path.name.lower(),
    )
    if not pdf_files:
        logger.warning(f"No PDF found in {folder}")
        return []
    
    docs = []
    for pdf in pdf_files:
        logger.info(f"Loading: {pdf.name}")
        try:
            docs.extend(PyPDFLoader(str(pdf)).load())
        except Exception as error:
            logger.warning(
                "pypdf could not parse %s (%s). Retrying with PyMuPDF.",
                pdf.name,
                error,
            )
            try:
                docs.extend(PyMuPDFLoader(str(pdf)).load())
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Both pypdf and PyMuPDF failed to parse {pdf.name}."
                ) from fallback_error
    
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

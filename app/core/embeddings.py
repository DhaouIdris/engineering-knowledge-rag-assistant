from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)
_embeddings = {}

def get_embeddings(model_name: str = None) -> HuggingFaceEmbeddings:
    selected_model = model_name or settings.embedding_model
    if selected_model not in _embeddings:
        logger.info(f"Loading embeddings model: {selected_model}")
        _embeddings[selected_model] = HuggingFaceEmbeddings(
            model_name=selected_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings[selected_model]

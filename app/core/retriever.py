from langchain_community.vectorstores import FAISS

from app.core.config import settings


def get_retriever(store: FAISS):
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": max(10, settings.retriever_k * 2),
        },
    )
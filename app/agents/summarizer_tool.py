from typing import List

from langchain.tools import Tool
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _format_documents(documents: List[Document]) -> str:
    """Transforme les documents récupérés en contexte textuel."""
    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "source inconnue")
        page = document.metadata.get("page")

        if page is not None:
            page_info = f", page {page + 1}"
        else:
            page_info = ""

        formatted_documents.append(
            f"[Document {index} - {source}{page_info}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_documents)


def create_summarizer_tool(retriever):
    """
    Crée un outil qui recherche les documents pertinents
    avant de les résumer.
    """

    llm = OllamaLLM(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
    )

    prompt = PromptTemplate.from_template(
        """
Tu es un assistant spécialisé dans la synthèse documentaire.

À partir du contexte fourni, produis un résumé fidèle en français.

Règles :
- Ne crée aucune information absente du contexte.
- Donne 3 à 7 points clés.
- Commence par une synthèse générale.
- Indique les sources utilisées à la fin.
- Si le contexte est insuffisant, dis-le explicitement.

Question de l'utilisateur :
{question}

Contexte documentaire :
{context}

Réponse :
"""
    )

    summarization_chain = prompt | llm

    def summarize_documents(query: str) -> str:
        """Recherche puis résume les documents pertinents."""
        logger.info("Recherche de documents pour résumé : %s", query)

        documents = retriever.invoke(query)

        if not documents:
            return (
                "Je n'ai trouvé aucun document pertinent pour produire "
                "un résumé."
            )

        context = _format_documents(documents)

        answer = summarization_chain.invoke(
            {
                "question": query,
                "context": context,
            }
        )

        sources = []
        for document in documents:
            source = document.metadata.get("source", "source inconnue")
            page = document.metadata.get("page")

            if page is not None:
                source_label = f"{source}, page {page + 1}"
            else:
                source_label = source

            if source_label not in sources:
                sources.append(source_label)

        source_text = "\n\nSources consultées :\n"
        source_text += "\n".join(f"- {source}" for source in sources)

        return f"{answer}{source_text}"

    return Tool(
        name="Summarizer",
        func=summarize_documents,
        description=(
            "À utiliser lorsqu'un utilisateur demande un résumé, une synthèse "
            "ou les principaux points des documents PDF. L'outil recherche "
            "automatiquement les passages pertinents dans FAISS avant de les "
            "résumer. L'entrée doit être la question ou le thème à résumer, "
            "par exemple : 'Résume les documents' ou "
            "'Quels sont les principaux sujets du corpus ?'."
        ),
    )
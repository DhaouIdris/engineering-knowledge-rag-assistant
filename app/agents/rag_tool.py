from langchain.tools import Tool
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_ollama import OllamaLLM

from app.core.config import settings


def create_rag_tool(retriever):
    llm = OllamaLLM(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    def search_documents(question: str) -> str:
        result = chain.invoke({"question": question})

        answer = result.get("answer", "")
        sources = result.get("sources", "")

        if sources:
            return f"{answer}\n\nSources : {sources}"

        return answer

    return Tool(
        name="DocumentSearch",
        func=search_documents,
        description=(
            "À utiliser pour répondre à une question précise sur le contenu "
            "des PDF. Recherche les passages pertinents dans FAISS et répond "
            "avec les sources. Ne pas utiliser pour faire un calcul ou un "
            "résumé général si l'outil Summarizer est plus adapté."
        ),
    )
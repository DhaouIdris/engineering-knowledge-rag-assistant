import streamlit as st
from pathlib import Path
from app.core.config import settings
from app.core.document_loader import load_documents, split_documents
from app.core.vector_store import build_vector_store, load_vector_store
from app.core.retriever import get_retriever
from app.core.memory import get_memory
from app.agents.agent_executor import create_agent
from app.core.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title=settings.app_name,
    page_icon="🤖",
    layout="wide"
)

# Load custom CSS
with open("app/ui/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = None
if "memory" not in st.session_state:
    st.session_state.memory = get_memory()
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    
    if st.button("📥 Indexer les documents", use_container_width=True):
        with st.spinner("Chargement et indexation des PDFs..."):
            docs = load_documents()
            chunks = split_documents(docs)
            store = build_vector_store(chunks)
            retriever = get_retriever(store)
            st.session_state.agent = create_agent(retriever, st.session_state.memory)
        st.success(f"✅ {len(docs)} pages indexées !")

    if st.button("📂 Charger index existant", use_container_width=True):
        with st.spinner("Chargement de l'index FAISS..."):
            store = load_vector_store()
            retriever = get_retriever(store)
            st.session_state.agent = create_agent(retriever, st.session_state.memory)
        st.success("✅ Index chargé !")

    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.session_state.memory = get_memory()
        st.rerun()

    st.markdown("---")
    st.markdown("**🛠️ Outils actifs :**")
    st.markdown("- 🔍 DocumentSearch (RAG)")
    st.markdown("- 🧮 Calculator")
    st.markdown("- 📋 Summarizer")

# ── Main Chat ──────────────────────────────────────────────────
st.title(f"🤖 {settings.app_name}")
st.caption(f"Propulsé par {settings.ollama_model} via Ollama • LangChain • FAISS")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Posez votre question sur les documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.agent is None:
        st.warning("⚠️ Veuillez d'abord indexer ou charger des documents.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                response = st.session_state.agent.invoke({"input": prompt})
                answer = response.get("output", "Je n'ai pas pu répondre.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
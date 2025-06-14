from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


LOG_DIR = "logs"
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/db_faiss")

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("course_assistant")


class OpenRouterLLM(LLM):
    """Minimal LangChain‑compatible wrapper around the OpenRouter Chat API."""

    api_key: str  # provided on instantiation
    model: str = "meta-llama/llama-4-scout:free"
    temperature: float = 0.35

    @property
    def _llm_type(self) -> str:  # noqa: D401
        "Return the identifier used by LangChain tooling (required)."
        return "openrouter"

    @property
    def _identifying_params(self) -> Dict[str, Any]:  # type: ignore[override]
        "LangChain expects a *mapping* here – not a method!"
        return {
            "model": self.model,
            "temperature": self.temperature,
        }

    def _call(self, prompt: str, stop: Optional[List[str]] | None = None) -> str:  # type: ignore[override]
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": prompt},
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer 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",
            "Content-Type": "application/json",
        }
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


@st.cache_resource(show_spinner="Initialisation des embeddings …")
def get_embeddings():
    logger.info("Loading HuggingFace embeddings …")
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
    )


@st.cache_resource(show_spinner="Chargement de la base vectorielle …")
def get_vectorstore():
    embeddings = get_embeddings()
    try:
        db = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Vectorstore loaded from %s", VECTORSTORE_PATH)
        return db
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load vectorstore: %s", exc)
        st.stop()


def collect_meta_values(db, key: str) -> List[str]:
    values: set[str] = set()
    for doc in db.docstore._dict.values():
        v = doc.metadata.get(key)
        if isinstance(v, str):
            values.add(v)
    return sorted(values)


def filter_docs(db, filters: Dict[str, List[str]]) -> FAISS:
    """Return a new FAISS store containing only docs that match *all* filters."""

    def keep(doc):  # noqa: ANN001
        for k, targets in filters.items():
            doc_val = doc.metadata.get(k)
            if doc_val is None:
                return False
            if isinstance(doc_val, list):
                if not any(t in doc_val for t in targets):
                    return False
            elif doc_val not in targets:
                return False
        return True

    docs = [d for d in db.docstore._dict.values() if keep(d)]
    if not docs:
        return db  # fall back to full DB if filter removes everything
    return FAISS.from_documents(docs, db._embedding_function)  # type: ignore[attr-defined]

@st.cache_resource
def french_course_prompt() -> PromptTemplate:
    template = (
        "Tu es un assistant pédagogique intelligent capable d'adapter tes réponses "
        "au niveau de détail souhaité par l'étudiant.\n\n"
        
        "CONTEXTE DES COURS:\n{context}\n\n"
        
        "QUESTION: {question}\n\n"
        
        "PROCESSUS DE RÉPONSE:\n"
        "1. ANALYSE: Détermine si la question demande:\n"
        "   - Une définition courte (ex: 'c'est quoi...?')\n"
        "   - Une explication moyenne (ex: 'comment...?')\n"
        "   - Une explication détaillée (ex: 'expliquez en détail...')\n"
        "   - CONTRAINTE DE LONGUEUR EXPLICITE (ex: 'en 1 ligne', 'en 3 lignes')\n"

        "2. RÉPONSE ADAPTÉE:\n"
        "   - Définition courte → 1-2 phrases maximum\n"
        "   - Explication moyenne → 1 paragraphe court\n"
        "   - Explication détaillée → Structure organisée\n"
        "   - '1 ligne' ou 'seul ligne' → MAXIMUM 15 mots\n"
        "   - '3 lignes' → EXACTEMENT 3 phrases courtes\n"
        "   - 'en X lignes' → RESPECTER STRICTEMENT le nombre demandé\n"

        "3. RÈGLES:\n"
        "   - Respecte le niveau de complexité demandé\n"
        "   - PRIORITÉ ABSOLUE aux contraintes de longueur explicites\n"
        "   - N'ajoute pas d'informations non sollicitées\n"
        "   - Base-toi uniquement sur le contexte\n"
        "   - Réponds en français\n"
        "   - Ne cite jamais les sources dans ta réponse\n\n"

        "RÉPONSE (adaptée à la complexité demandée):"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])


def retrieve_relevant_docs(db, user_query: str, embeddings, threshold: float = 0.35, top_k: int = 5):
    query_embedding = np.array(embeddings.embed_query(user_query))
    
    documents = list(db.docstore._dict.values())
    doc_embeddings = db.index.reconstruct_n(0, db.index.ntotal)
    
    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    dot_products = np.dot(doc_embeddings, query_embedding)
    cosine_similarities = dot_products / (doc_norms * query_norm + 1e-10)  # Avoid division by zero
    
    doc_sim_pairs = [(doc, sim) for doc, sim in zip(documents, cosine_similarities)]
    doc_sim_pairs.sort(key=lambda x: x[1], reverse=True)
    
    selected_docs = []
    for doc, sim in doc_sim_pairs:
        if sim >= threshold and len(selected_docs) < top_k:
            # Add similarity score to metadata for debugging/inspection
            doc.metadata["similarity_score"] = float(sim)
            selected_docs.append(doc)
    
    return selected_docs

class CustomRetriever(BaseRetriever):
    """Custom retriever that returns pre-filtered documents."""
    
    documents: List[Document]
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents


def main() -> None:  # noqa: C901 – easiest single‑file entry‑point
    st.set_page_config(page_title="Assistant de Cours", page_icon="📚", layout="wide")
    st.title("📚 Assistant de Cours Universitaire")

    db = get_vectorstore()
    modules = collect_meta_values(db, "module")
    doc_types = collect_meta_values(db, "document_type")

    with st.sidebar:
        st.header("🔎 Filtres de recherche")
        sel_modules = st.multiselect("Modules", modules)
        sel_doc_types = st.multiselect("Types de document", doc_types)
        
        st.header("⚙️ Paramètres de recherche")
        similarity_threshold = st.slider(
            "Seuil de similarité", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.35,
            step=0.05,
            help="Augmentez pour des résultats plus précis, diminuez pour plus de documents"
        )
        max_docs = st.slider(
            "Nombre max. de documents", 
            min_value=1, 
            max_value=20, 
            value=5,
            help="Nombre maximum de documents à récupérer"
        )

    filters: Dict[str, List[str]] = {}
    if sel_modules:
        filters["module"] = sel_modules
    if sel_doc_types:
        filters["document_type"] = sel_doc_types

    messages = []  # Initialize messages list
    
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Posez une question sur vos cours …"):
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion …"):
                query_id = uuid.uuid4()
                search_db = filter_docs(db, filters) if filters else db

                llm = OpenRouterLLM(
                    api_key=os.getenv("OPENROUTER_API_KEY", ""),
                    model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-4-scout:free"),
                )

                relevant_docs = retrieve_relevant_docs(
                    search_db, 
                    prompt, 
                    get_embeddings(),
                    threshold=similarity_threshold,
                    top_k=max_docs
                )
                
                # Create a custom retriever with pre-filtered documents
                retriever = CustomRetriever(documents=relevant_docs)
                
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": french_course_prompt(),
                    },
                )

                try:
                    start = time.perf_counter()
                    # Use the standard query format
                    result = qa({"query": prompt})
                    elapsed = time.perf_counter() - start

                    answer: str = result["result"]

                    st.markdown(answer)
                    messages.append({"role": "assistant", "content": answer})
                    logger.info("Answered id=%s in %.2fs", query_id, elapsed)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("QA failure: %s", exc)
                    err = f"🚨 Erreur lors du traitement de votre requête : {exc}"
                    st.error(err)
                    messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()

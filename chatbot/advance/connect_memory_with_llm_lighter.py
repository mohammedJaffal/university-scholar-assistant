from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


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
    temperature: float = 0.7

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
            "Authorization": f"Bearer {self.api_key}",
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
        "Tu es un assistant intelligent spécialisé pour répondre aux questions "
        "relatives aux cours universitaires. Utilise le contexte suivant pour répondre à la "
        "question de l'utilisateur.\n\n"
        "Contexte:\n{context}\n\n"
        "Question: {question}\n\n"
        "Instructions spécifiques:\n"
        "- Réponds en français uniquement\n"
        "- Cite les sources pertinentes en fin de réponse\n"
        "- Structure ta réponse avec des titres si nécessaire\n"
        "- Réponds uniquement si l'information est dans le contexte\n"
        "- Si aucune information pertinente n'est disponible, indique‑le clairement\n\n"
        "Réponse:"  # model continues here
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])


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
        k = st.slider("Nombre de passages", min_value=3, max_value=10, value=5)
        hide_sources = st.checkbox("Masquer les sources", value=False)

    filters: Dict[str, List[str]] = {}
    if sel_modules:
        filters["module"] = sel_modules
    if sel_doc_types:
        filters["document_type"] = sel_doc_types

    messages = st.session_state.setdefault("messages", [])
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

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=search_db.as_retriever(search_kwargs={"k": k}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": french_course_prompt()},
                )

                try:
                    start = time.perf_counter()
                    result = qa(prompt)
                    elapsed = time.perf_counter() - start

                    answer: str = result["result"]
                    sources = result.get("source_documents", [])

                    if sources and not hide_sources:
                        lines = {s.metadata.get("source", f"Document {i+1}") for i, s in enumerate(sources)}
                        answer += "\n\n**Sources :**\n" + "\n".join(f"- {s}" for s in sorted(lines))

                    st.markdown(answer)
                    messages.append({"role": "assistant", "content": answer})
                    logger.info("Answered id=%s in %.2fs", query_id, elapsed)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("QA failure: %s", exc)
                    err = f"🚨 Erreur lors du traitement de votre requête : {exc}"
                    st.error(err)
                    messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":  # pragma: no cover
    main()

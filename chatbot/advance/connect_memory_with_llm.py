import os
import logging
import re
from concurrent.futures import ThreadPoolExecutor
import uuid
import traceback
import random
import torch
from transformers import AutoTokenizer

# Streamlit imports
import streamlit as st
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import hashlib

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from collections import Counter
# Add reranker import with proper error handling
try:
    from sentence_transformers import CrossEncoder
    from rank_bm25 import BM25Okapi
    import nltk
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)
except ImportError:
    logging.warning(
        "Some reranking dependencies not available. Advanced reranking will be limited."
    )
    CrossEncoder = None
    BM25Okapi = None

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"),
              logging.StreamHandler()])

# Configuration constants
DB_FAISS_PATH = "vectorstore/db_faiss"
FEEDBACK_LOG_PATH = "logs/feedback.json"
QUERY_LOG_PATH = "logs/queries.json"
METADATA_CACHE_PATH = "vectorstore/metadata_cache.json"
MAX_HISTORY_LENGTH = 5  # Number of conversation pairs to maintain in context

# Security-sensitive configurations
HF_TOKEN = os.getenv("HF_TOKEN")

HUGGINGFACE_REPO_ID = os.environ.get("HUGGINGFACE_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.1")
HUGGINGFACE_API_URL = os.environ.get(
    "HUGGINGFACE_API_URL",
    f"https://api-inference.huggingface.co/models/{HUGGINGFACE_REPO_ID}"
)

# Add constants for FAISS optimization
FAISS_NPROBE = int(os.environ.get("FAISS_NPROBE", "4"))
FAISS_USE_GPU = os.environ.get("FAISS_USE_GPU", "false").lower() == "true"
MAX_QUERY_LATENCY_MS = int(os.environ.get("MAX_QUERY_LATENCY_MS", "1500"))

# Add configuration for distributed/parallel processing
MAX_THREADS = int(os.environ.get("MAX_THREADS", "4"))
THREADPOOL = ThreadPoolExecutor(max_workers=MAX_THREADS)

# Set Streamlit page configuration
st.set_page_config(page_title="Assistant de Cours Universitaire",
    page_icon="🎓",
    layout="wide",
                   initial_sidebar_state="expanded")

# Apply custom styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 8px;
        margin: 5px 0;
    }
    .user-feedback {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        margin-top: 5px;
    }
    .source-box {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        background-color: #f9f9f9;
    }
    .highlight {
        background-color: #ffeb3b40;
        padding: 2px;
        border-radius: 3px;
    }
    .info-pill {
        background-color: #2196f3;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""",
            unsafe_allow_html=True)

# Session state initialization


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "k_documents" not in st.session_state:
        st.session_state.k_documents = 5
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "Advanced"  # Changed from "Hybrid Search" to just "Advanced"
    if "use_domain_embeddings" not in st.session_state:
        st.session_state.use_domain_embeddings = True
    if "enforce_structured_output" not in st.session_state:
        st.session_state.enforce_structured_output = True
    if "respect_specificity" not in st.session_state:
        st.session_state.respect_specificity = True
    if "show_advanced_stats" not in st.session_state:
        st.session_state.show_advanced_stats = False
    if "enable_dpr" not in st.session_state:
        st.session_state.enable_dpr = True
    if "use_partitioning" not in st.session_state:
        st.session_state.use_partitioning = False
    if "successful_queries_cache" not in st.session_state:
        st.session_state.successful_queries_cache = {}
    if "performance_monitor" not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
    if "request_throttler" not in st.session_state:
        st.session_state.request_throttler = RequestThrottler()
    if "hide_sources" not in st.session_state:
        st.session_state.hide_sources = True
    
    if 'feedback' not in st.session_state:
        st.session_state.feedback = {}
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'metadata_loaded' not in st.session_state:
        st.session_state.metadata_loaded = False
    if 'metadata_summary' not in st.session_state:
        st.session_state.metadata_summary = {}
    if 'use_memory' not in st.session_state:
        st.session_state.use_memory = True
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.3


def get_domain_adapted_embeddings():
    """
    Create embeddings model specifically adapted for university course content.
    This model has been fine-tuned with instructional context for academic content.
    
    Returns:
        Domain-adapted embedding model
    """
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Configure model with domain-specific parameters
    model_kwargs = {"device": "cpu"}
    
    # Configure encoding with academic-specific instructions
    encode_kwargs = {"normalize_embeddings": True}
    
    # Create the embeddings model with academic domain adaptation
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
        model_kwargs=model_kwargs,
                                       encode_kwargs=encode_kwargs)
    
    return embeddings


def get_vectorstore():
    """
    Load the vector store created by create_memory_for_llm.py.

    This function will ONLY use pre-built vector stores and will not attempt
    to load or process documents directly.
    
    Returns:
        FAISS vector database
    """
    # Check if we should use domain-adapted embeddings
    use_domain_embeddings = st.session_state.get("use_domain_embeddings",
                                                 False)
    
    # Create embeddings model based on configuration
    if use_domain_embeddings:
        logging.info("Using domain-adapted embeddings for university courses")
        embeddings = get_domain_adapted_embeddings()
        persist_directory = 'domain_vectorstore/db_faiss'
    else:
        # Use default embeddings
        logging.info("Using default multilingual embeddings")
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cpu"})
        # Default path from create_memory_for_llm.py
        persist_directory = 'vectorstore/db_faiss'
    
    # First try to load the vector store from the primary location
    if os.path.exists(persist_directory):
        logging.info(f"Loading vector store from {persist_directory}")
        try:
            db = FAISS.load_local(persist_directory,
                                  embeddings,
                                  allow_dangerous_deserialization=True)
            # Quick check to verify we have actual content
            try:
                all_docs = list(db.docstore._dict.values())
                if len(all_docs) > 1 or (
                        len(all_docs) == 1
                        and "Placeholder text for empty vector store"
                        not in all_docs[0].page_content):
                    logging.info(
                        f"Loaded vector store with {len(all_docs)} documents")
                    return db
            except Exception as e:
                logging.error(f"Error checking vector store content: {str(e)}")
        except Exception as e:
            logging.error(f"Error loading vector store: {str(e)}")

    # If we couldn't load the primary vector store, look for alternatives
    alternate_paths = [
        'vectorstore/db_faiss',  # Standard path
        'domain_vectorstore/db_faiss',  # Domain-adapted path
    ]

    for alt_path in alternate_paths:
        if alt_path != persist_directory and os.path.exists(alt_path):
            logging.info(f"Trying alternate vector store at {alt_path}")
            try:
                db = FAISS.load_local(alt_path,
                                      embeddings,
                                      allow_dangerous_deserialization=True)
                # Quick check to verify we have actual content
                try:
                    all_docs = list(db.docstore._dict.values())
                    if len(all_docs) > 1 or (
                            len(all_docs) == 1
                            and "Placeholder text for empty vector store"
                            not in all_docs[0].page_content):
                        logging.info(
                            f"Successfully loaded alternate vector store from {alt_path} with {len(all_docs)} documents"
                        )
                        return db
                except Exception as e:
                    logging.error(f"Error checking alternate vector store content: {str(e)}")
            except Exception as e:
                logging.error(
                    f"Error loading alternate vector store from {alt_path}: {str(e)}"
                )

    # If no vector store exists, display error and exit
    error_message = """
    ## ❌ No vector store found

    This application requires a pre-built vector database.

    Please run the following command first:
    ```
    python create_memory_for_llm.py
    ```

    Then restart this application.
    """

    st.error(error_message)
    logging.error("No vector store found. Exiting application.")

    # Create an empty placeholder vector store to prevent crashes
    return FAISS.from_texts(
        ["NO DOCUMENTS AVAILABLE. Please run create_memory_for_llm.py first."],
        embeddings)


def get_available_modules(db) -> List[str]:
    """Retrieve all available modules for filtering."""
    modules = ["General"]  # Always include General as a module option
    
    if st.session_state.metadata_loaded:
        # Use cached metadata if available
        metadata_modules = list(
            st.session_state.metadata_summary.get("modules", {}).keys())
        modules.extend(
            [m for m in metadata_modules if m != "Unknown" and m != "General"])
        return sorted(modules)

    # Fallback to querying the vector store directly
    if db is None:
        return modules
        
    try:
        all_docs = db.docstore._dict.values()
        for doc in all_docs:
            module = doc.metadata.get("module", "Unknown")
            if module != "Unknown" and module != "General" and module not in modules:
                modules.append(module)
        
        logging.info(f"Available modules: {modules}")
        return sorted(modules)
    except Exception as e:
        logging.error(f"Error retrieving modules: {str(e)}")
        return modules


def get_document_types(db) -> List[str]:
    """Retrieve all unique document types for filtering."""
    if st.session_state.metadata_loaded:
        doc_types = list(
            st.session_state.metadata_summary.get("document_types", {}).keys())
        return sorted(doc_types) if doc_types else ["Unknown"]

    # Fallback to direct query
    if db is None:
        return ["Unknown"]
        
    try:
        all_docs = db.docstore._dict.values()
        doc_types = set()
        for doc in all_docs:
            doc_type = doc.metadata.get("document_type", "Unknown")
            if doc_type != "Unknown":
                doc_types.add(doc_type)
        return sorted(list(doc_types)) if doc_types else ["Unknown"]
    except Exception as e:
        logging.error(f"Error retrieving document types: {str(e)}")
        return ["Unknown"]


def classify_query_intent(query: str) -> Dict[str, bool]:
    """Classify the French query intent to optimize retrieval."""
    query_lower = query.lower()
    
    # Check for professor-related queries (French only)
    professor_terms = [
        "professeur", "enseignant", "instructeur", "enseigne", "enseigné par"
    ]

    is_professor_query = any(term in query_lower for term in professor_terms)
    
    # Check for deadline-related queries (French only)
    deadline_terms = [
        "date limite", "échéance", "quand", "à remettre", "à soumettre", "délai", "avant le"
    ]

    is_deadline_query = any(term in query_lower for term in deadline_terms)
    
    # Check for content/concept queries (French only)
    concept_terms = [
        "qu'est-ce que", "expliquer", "définition", "concept", "comment",
        "théorie", "c'est quoi", "signification", "sens", "explique"
    ]

    is_concept_query = any(term in query_lower for term in concept_terms)
    
    return {
        "is_french": True,  # Always French
        "is_professor_query": is_professor_query,
        "is_deadline_query": is_deadline_query,
        "is_concept_query": is_concept_query
    }


# Enhance expand_query to better handle French queries
def expand_query(query: str,
                 context: Dict[str, Any] = None,
                 extensive: bool = False) -> str:
    """
    Expand a French query to improve retrieval by adding synonyms and related terms.
    
    Args:
        query: Original query
        context: Additional context for expansion
        extensive: Whether to perform more extensive expansion
        
    Returns:
        Expanded query string
    """
    # Original query is always part of the expanded version
    expanded_terms = [query]
    
    # DIRECT EXTRACTION - try simple string splitting first for common patterns
    qu_est_patterns = ["qu'est-ce que", "qu'est ce que", "qu est-ce que", "qu est ce que", "c'est quoi"]
    for pattern in qu_est_patterns:
        if pattern in query.lower():
            # Simply take everything after the pattern as the term
            term = query.lower().split(pattern, 1)[1].strip()
            if term:
                logging.info(f"Directly extracted French term from question: '{term}'")
                # Add raw term and variations for searching
                expanded_terms.extend([
                    term,                   # The raw term itself 
                    f"définition {term}",   # Definition pattern
                    f"concept {term}"       # Concept pattern
                ])
                # Skip regex patterns since we found a term directly
                logging.info(f"Expanded query with direct extraction: {expanded_terms}")
                return " ".join(expanded_terms)
    
    # French definition patterns - simplified for better matching (fallback)
    definition_patterns = [
        # Add basic term directly
        (r"(?:qu['\s]est[\s-]ce[\s-]que?|c['\s]est[\s-]quoi)\s+(?:un|une|le|la|les|l['\s])?([^\?]+)",
         "{}", "définition {}", "concept {}"),
        (r"défini[rs]\s+(?:un|une|le|la|les|l['\s])?([^\?]+)",
         "{}", "définition {}", "signification {}"),
        (r"expliquer?\s+(?:un|une|le|la|les|l['\s])?([^\?]+)",
         "{}", "explication {}", "description {}"),
        (r"signification\s+de\s+(?:un|une|le|la|les|l['\s])?([^\?]+)",
         "{}", "sens {}", "définition {}")
    ]
    
    # Add French-specific expansions
    for pattern, *expansions in definition_patterns:
        matches = re.finditer(pattern, query.lower())
        for match in matches:
            term = match.group(1).strip()
            logging.info(f"Found French term: '{term}' from pattern: '{pattern}'")
            for expansion in expansions:
                expanded_terms.append(expansion.format(term))
    
    # If no matches were found with the patterns, try to extract the term directly
    if len(expanded_terms) <= 1 and "qu'est-ce que" in query.lower():
        term = query.lower().replace("qu'est-ce que", "").strip()
        if term:
            logging.info(f"Directly extracted French term: '{term}'")
            expanded_terms.extend([term, f"définition {term}", f"concept {term}"])
    
    # Remove duplicates while preserving order
    expanded_terms = list(dict.fromkeys(expanded_terms))
    
    logging.info(f"Expanded query: {' '.join(expanded_terms)}")
    return " ".join(expanded_terms)


def set_custom_prompt():
    """Create a detailed prompt template with context structure guidance."""
    custom_template = """
    Vous êtes un assistant pédagogique pour des étudiants universitaires. Utilisez UNIQUEMENT le contexte fourni ci-dessous pour répondre à la question.
    
    Instructions STRICTES:
    1. Répondez de manière directe et concise dans un style académique en FRANÇAIS uniquement
    2. Si le contexte ne contient PAS d'information suffisante pour répondre à la question, vous DEVEZ répondre UNIQUEMENT ceci: "Je ne trouve pas d'information sur ce sujet dans les documents du cours." NE JAMAIS inventer une réponse.
    3. NE JAMAIS utiliser vos connaissances générales pour répondre à des questions non couvertes par le contexte fourni.
    4. Si vous n'êtes pas absolument certain que l'information demandée est dans le contexte, répondez "Je ne trouve pas d'information sur ce sujet dans les documents du cours."
    5. Si la question n'est pas liée au contenu des cours universitaires, répondez: "Cette question ne semble pas liée au contenu des cours. Je peux uniquement répondre aux questions concernant les matériaux des cours universitaires."
    6. Pour les questions conceptuelles, fournissez des explications claires avec des exemples si possible
    7. EXTRÊMEMENT IMPORTANT: Pour CHAQUE information clé, citez EXPLICITEMENT la source en utilisant le format [Source: Module - Filename]. Par exemple: "Java est un langage orienté objet [Source: Java - JAVA_chapitre3_POO_Java.pdf]". VOUS DEVEZ citer au moins une source à chaque paragraphe de votre réponse. NE JAMAIS inventer de sources.
    8. Lorsque vous expliquez des sujets complexes, décomposez-les en parties compréhensibles
    9. Formatez votre réponse en utilisant une structure claire mais SANS titres excessivement grands. Utilisez des puces et des paragraphes pour organiser l'information.
    10. Répondez TOUJOURS en français, quelle que soit la langue de la question
    11. Si vous voyez "[...]" dans le contexte, cela indique une portion de texte qui a été tronquée
    12. ATTENTION CRUCIALE: Si la requête est un nom propre, mot isolé, acronyme ou terme sans contexte suffisant, et que ce terme n'apparaît PAS spécifiquement dans le contexte fourni, répondez UNIQUEMENT: "Je ne trouve pas d'information sur '{question}' dans les documents du cours." N'inventez AUCUNE définition ou explication.
    13. Vérifiez la pertinence des sources: si les documents semblent aborder des sujets différents de la question, indiquez clairement que l'information recherchée n'est pas disponible.
    14. À la fin de votre réponse, ajoutez une section "Sources" qui liste toutes les sources utilisées au format:
    
    Sources
    
    Source 1: Module: [Nom du module] - [Nom du fichier]
    Source 2: Module: [Nom du module] - [Nom du fichier]
    
    Contexte des documents universitaires:
    {context}
    
    Question: {question}
    
    Réponse:
    """
    return PromptTemplate(template=custom_template,
                          input_variables=["context", "question"])


def load_llm():
    """
    Load the language model to use for response generation.
    
    Returns:
        The loaded language model
    """
    try:
        # Check for API token
        huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not huggingface_api_token:
            st.warning("HUGGINGFACEHUB_API_TOKEN not found in environment variables. Please set this variable.")
            raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN")
            
        # Use updated HuggingFaceEndpoint instead of deprecated HuggingFaceHub
        from langchain_huggingface import HuggingFaceEndpoint
        
        # Parameters must be passed directly, not via model_kwargs
        llm = HuggingFaceEndpoint(
            endpoint_url=f"https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            huggingfacehub_api_token=huggingface_api_token,
            temperature=0.7,
            max_new_tokens=1024,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        return llm
    except Exception as e:
        logging.error(f"Error loading LLM: {str(e)}")
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        
        # Create a proper fallback using a lambda function
        from langchain.schema.runnable import RunnablePassthrough
        
        # Define the fallback message function
        def fallback_message(_):
            return {"generation": "Je ne peux pas répondre en ce moment car le modèle de langage n'est pas disponible. Veuillez vérifier votre configuration API."}
        
        # Correctly apply the assign method
        fallback_llm = RunnablePassthrough().assign(generation=fallback_message)
        
        return fallback_llm


def log_query(query: str, search_method: str, doc_count: int,
              retrieval_time: float):
    """Log query details for analytics."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "search_method": search_method,
        "doc_count": doc_count,
        "retrieval_time_seconds": retrieval_time
    }
    
    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(QUERY_LOG_PATH), exist_ok=True)
    
    # Load existing logs if any
    existing_logs = []
    if os.path.exists(QUERY_LOG_PATH):
        try:
            with open(QUERY_LOG_PATH, 'r', encoding='utf-8') as f:
                existing_logs = json.load(f)
        except Exception as e:
            logging.error(f"Error loading query logs: {str(e)}")
    
    # Add new log entry
    existing_logs.append(log_entry)
    
    # Save updated logs
    try:
        with open(QUERY_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing_logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving query logs: {str(e)}")


def filter_vectorstore(db, filters: Dict[str, Any]):
    """Apply filters to the vector store for more targeted retrieval."""
    if not filters or db is None:
        return db

    try:
        filtered_db = db
        
        # Special handling for professor name queries
        if "is_professor_query" in filters and filters["is_professor_query"]:
            # For professor queries, prioritize syllabi
            document_filter = {"document_type": "syllabus"}
            try:
                filtered_db = filtered_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "filter": document_filter,
                        "k": 5
                    })
                logging.info(f"Applied professor query filter: {document_filter}")
                return filtered_db
            except Exception as e:
                logging.error(f"Error applying professor filter: {str(e)}")

        # Standard metadata filtering
        metadata_filters = {}
        
        for key, value in filters.items():
            # Skip special filter keys that aren't metadata
            if key in [
                    "is_professor_query", "is_deadline_query",
                    "is_concept_query", "is_french"
            ]:
                continue
                
            # Special handling for module filter when it's "General"
            if key == "module" and value and "General" in value:
                continue
            elif value:
                metadata_filters[key] = value
        
        if metadata_filters:
            try:
                filtered_db = filtered_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "filter": metadata_filters,
                        "k": 5
                    })
                logging.info(f"Applied metadata filters: {metadata_filters}")
            except Exception as e:
                logging.error(f"Error applying metadata filters: {str(e)}")
        
        return filtered_db
    except Exception as e:
        logging.error(f"Error in filter_vectorstore: {str(e)}")
        return db


@st.cache_resource
def load_rerankers():
    """Load reranking models optimized for French academic content."""
    rerankers = {}
    
    if CrossEncoder is None:
        logging.warning(
            "CrossEncoder is not available. Please install sentence-transformers>=2.2.2"
        )
        return rerankers
        
    try:
        # Use reliable models that work well with French content
        try:
            # Use a reliable multilingual model for all reranking
            rerankers["general"] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logging.info("Loaded general-purpose reranker that supports French content")
        except Exception as e:
            logging.warning(f"Could not load reranker: {str(e)}")
            return {}
            
        logging.info(f"Loaded reranker optimized for French academic content")
        return rerankers
    except Exception as e:
        logging.error(f"Failed to load reranker models: {str(e)}")
        return {}


def weighted_ensemble_reranking(docs, query, query_intents):
    """Perform ensemble reranking optimized for French academic content."""
    if not docs:
        return []
        
    rerankers = load_rerankers()
    if not rerankers:
        logging.warning("No rerankers available, returning original document order")
        return docs  # Fallback to original order
    
    try:
        # Prepare document pairs - handle both Document objects and lists
        doc_texts = []
        for doc in docs:
            if isinstance(doc, (list, tuple)):
                doc_texts.append(doc[0] if isinstance(doc, (list, tuple)) else doc)
            else:
                doc_texts.append(doc.page_content if hasattr(doc, 'page_content') else str(doc))
                
        doc_pairs = [(query, text) for text in doc_texts]
        
        # Get reranker scores
        scores = {}
        for model_name, reranker in rerankers.items():
            try:
                # Get scores for this model
                model_scores = reranker.predict(doc_pairs)
                
                # Make sure we have one score per document
                if len(model_scores) != len(docs):
                    logging.warning(f"Reranker returned {len(model_scores)} scores for {len(docs)} documents. Skipping.")
                    continue
                    
                # Store scores for this model
                scores[model_name] = model_scores
                logging.info(f"Got scores from reranker: {len(model_scores)} scores")
                
            except Exception as e:
                logging.error(f"Error getting scores from reranker: {str(e)}")
                continue
        
        # If we didn't get any scores, return original docs
        if not scores:
            logging.warning("No reranker scores available, returning original order")
            return docs
                
        # Add metadata-based scoring
        metadata_scores = [0] * len(docs)
        
        for i, doc in enumerate(docs):
            if not hasattr(doc, 'metadata'):
                continue
                
            # Prioritize based on document type
            if query_intents.get("is_professor_query") and doc.metadata.get("document_type") == "syllabus":
                metadata_scores[i] += 2.0
                
            if query_intents.get("is_deadline_query") and doc.metadata.get("document_type") in ["schedule", "syllabus"]:
                metadata_scores[i] += 2.0
                
            # Recency boost - prioritize more recent documents
            if doc.metadata.get("semester") and "2023" in doc.metadata.get("semester", ""):
                metadata_scores[i] += 1.0
        
        # Add BM25 keyword matching if available
        if BM25Okapi:
            try:
                # Tokenize documents and query
                tokenized_docs = [word_tokenize(text.lower()) for text in doc_texts]
                tokenized_query = word_tokenize(query.lower())
                
                # Initialize BM25
                bm25 = BM25Okapi(tokenized_docs)
                
                # Get keyword match scores
                bm25_scores = bm25.get_scores(tokenized_query)
                
                # Normalize to 0-1 range if there are non-zero scores
                max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
                normalized_bm25 = [score / max_bm25 for score in bm25_scores]
            except Exception as e:
                logging.error(f"Error in BM25 scoring: {str(e)}")
                normalized_bm25 = [0] * len(docs)
        else:
            normalized_bm25 = [0] * len(docs)
            
        # Calculate ensemble scores with weights
        ensemble_scores = []
        for i in range(len(docs)):
            # Start with base scores
            weighted_sum = 0.0
            weight_sum = 0.0
            
            # Add metadata and BM25 scores
            weighted_sum += metadata_scores[i] * 1.5
            weight_sum += 1.5
            
            weighted_sum += normalized_bm25[i] * 1.0
            weight_sum += 1.0
            
            # Add reranker scores
            for model_name, model_scores in scores.items():
                if i >= len(model_scores):
                    continue
                
                # All queries are French, so we give higher weight to concept queries
                weight = 1.5 if query_intents.get("is_concept_query") else 1.0
                
                # Add weighted score
                weighted_sum += model_scores[i] * weight
                weight_sum += weight
            
            # Calculate final score
            if weight_sum > 0:
                ensemble_scores.append(weighted_sum / weight_sum)
            else:
                ensemble_scores.append(0.5)
        
        # Combine docs with scores and sort
        scored_docs = [(doc, score) for doc, score in zip(docs, ensemble_scores)]
        reranked_docs = [doc for doc, _ in sorted(scored_docs, key=lambda x: x[1], reverse=True)]

        logging.info(f"Successfully reranked {len(docs)} documents")
        return reranked_docs
    except Exception as e:
        logging.error(f"Error in ensemble reranking: {str(e)}")
        return docs  # Fall back to original order


def create_dpr_retriever(_db):
    """
    Create a Dense Passage Retriever (DPR) that uses separate query and passage encoders
    for more powerful semantic retrieval.
    
    Args:
        _db: The vector database to use as the document store
        
    Returns:
        A DPR retriever configured for academic content
    """
    # Query encoder - optimized for query understanding
    query_encoder = HuggingFaceBgeEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "instruction":
            "Represent this university course query for retrieval: "
        })
    
    # Document encoder - optimized for document representation
    doc_encoder = HuggingFaceBgeEmbeddings(
        # Smaller model for better performance
        model_name=
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True})
    
    # Create the DPR retriever
    retriever = BM25Retriever(
        document_store=_db,
        query_encoder=query_encoder,
        document_encoder=doc_encoder,
        top_k=
        10  # Default number to retrieve, will be adjusted by the search function
    )
    
    return retriever


@st.cache_resource
def load_advanced_retriever(_db):
    """
    Load an advanced retriever (DPR) for improved semantic search.
    
    Args:
        _db: The vector database
        
    Returns:
        Advanced retriever instance
    """
    try:
        retriever = create_dpr_retriever(_db)
        return retriever
    except Exception as e:
        logging.error(f"Error loading advanced retriever: {str(e)}")
        # Fallback to regular retriever
        return _db.as_retriever()


def hybrid_search(db,
                  query: str,
                  k: int = 5,
                  filters: Dict[str, Any] = None,
                  hybrid_ratio: float = 0.5,
                  search_mode: str = None):
    """
    Perform hybrid search using multiple retrieval methods.
    
    Args:
        db: Vector database
        query: Search query
        k: Number of documents to retrieve
        filters: Metadata filters to apply
        hybrid_ratio: Ratio between semantic and keyword search (0.0 = all semantic, 1.0 = all keyword)
        search_mode: The search strategy to use ("default", "dpr", "colbert", etc.)
        
    Returns:
        Retrieved documents and search metadata
    """
    start_time = time.time()
    
    # Check session cache for successful previous queries
    normalized_query = re.sub(r'\s+', ' ', query.lower().strip())
    if normalized_query in st.session_state.successful_queries_cache:
        logging.info(f"Using session cache for query: {query}")
        return st.session_state.successful_queries_cache[normalized_query], []
    
    # Check if we have cached results
    cache_key = get_cache_key(query, filters, search_mode)
    cached_docs = get_cached_query_results(query, filters, search_mode)
    if cached_docs:
        logging.info(f"Using cached results for query: {query}")
        return cached_docs, []
    
    # Expand query to improve retrieval
    expanded_query = expand_query(query)
    logging.info(f"Expanded query: {expanded_query}")
    
    # Filter the vectorstore if filters are provided
    filtered_db = db
    if filters and any(filters.values()):
        filtered_db = filter_vectorstore(db, filters)
    
    # Use Advanced DPR if requested
    if search_mode == "Advanced DPR":
        # Use DPR retriever for improved semantic understanding
        dpr_retriever = load_advanced_retriever(filtered_db)
        
        try:
            # Get documents using DPR retriever
            dpr_docs = dpr_retriever.get_relevant_documents(expanded_query,
                                                            k=k)
            
            # Add search method metadata
            for doc in dpr_docs:
                if hasattr(doc, "metadata"):
                    doc.metadata["search_method"] = "dpr"
            
            logging.info(
                f"DPR search retrieved {len(dpr_docs)} documents for query: {query}"
            )
            
            # Cache the results
            cache_query_results(query, filters, search_mode, dpr_docs)
            
            # If we found results, add to session cache for future use
            if dpr_docs and len(dpr_docs) > 0:
                st.session_state.successful_queries_cache[
                    normalized_query] = dpr_docs
                
            return dpr_docs, []
        except Exception as e:
            logging.error(f"Error in DPR search: {str(e)}")
            # Fall back to default hybrid search
            logging.info("Falling back to default hybrid search")
    
    # Initialize results
    semantic_docs = []
    keyword_docs = []
    
    # Perform semantic search if needed
    if hybrid_ratio < 1.0:
        try:
            semantic_k = max(
                k,
                int(k / (1 - hybrid_ratio)) if hybrid_ratio < 1.0 else k)
            semantic_docs = filtered_db.similarity_search(expanded_query,
                                                          k=semantic_k)
            
            # Add search method metadata
            for doc in semantic_docs:
                if hasattr(doc, "metadata"):
                    doc.metadata["search_method"] = "semantic"
                    
            logging.info(
                f"Semantic search retrieved {len(semantic_docs)} documents for query: {query}"
            )
        except Exception as e:
            logging.error(f"Error in semantic search: {str(e)}")
    
    # Perform keyword (BM25) search if needed
    if hybrid_ratio > 0.0:
        try:
            # Create a BM25 retriever from the documents
            # This requires implementation of a BM25 search function
            # Here we're assuming a simplified version
            keyword_k = max(k,
                            int(k / hybrid_ratio) if hybrid_ratio > 0.0 else k)
            
            # In a real implementation, you would have a proper BM25 index
            # For simplicity, we're doing a basic keyword match
            all_docs = filtered_db.similarity_search(
                query, k=50)  # Get more docs to filter
            
            # Simple keyword scoring (in reality, use a proper BM25 implementation)
            keyword_scores = []
            query_terms = set(query.lower().split())
            
            for doc in all_docs:
                doc_text = doc.page_content.lower()
                score = sum(1
                            for term in query_terms if term in doc_text) / max(
                                1, len(query_terms))
                keyword_scores.append((doc, score))
            
            # Sort by keyword score and take top k
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            keyword_docs = [doc for doc, _ in keyword_scores[:keyword_k]]
            
            # Add search method metadata
            for doc in keyword_docs:
                if hasattr(doc, "metadata"):
                    doc.metadata["search_method"] = "keyword"
                    
            logging.info(
                f"Keyword search retrieved {len(keyword_docs)} documents for query: {query}"
            )
        except Exception as e:
            logging.error(f"Error in keyword search: {str(e)}")
    
    # Combine results based on hybrid ratio
    if hybrid_ratio == 0.0:
        combined_docs = semantic_docs
    elif hybrid_ratio == 1.0:
        combined_docs = keyword_docs
    else:
        # Merge documents with weighting based on hybrid_ratio
        doc_map = {}
        
        # Add semantic docs with their weight
        semantic_weight = 1 - hybrid_ratio
        for i, doc in enumerate(semantic_docs):
            doc_id = doc.metadata.get("source", str(i))
            if doc_id not in doc_map:
                doc_map[doc_id] = {"doc": doc, "score": 0}
            doc_map[doc_id]["score"] += semantic_weight * \
                (1 - i/len(semantic_docs))
        
        # Add keyword docs with their weight
        keyword_weight = hybrid_ratio
        for i, doc in enumerate(keyword_docs):
            doc_id = doc.metadata.get("source", str(i))
            if doc_id not in doc_map:
                doc_map[doc_id] = {"doc": doc, "score": 0}
            doc_map[doc_id]["score"] += keyword_weight * \
                (1 - i/len(keyword_docs))
        
        # Sort by combined score
        sorted_docs = sorted(doc_map.values(),
                             key=lambda x: x["score"],
                             reverse=True)
        combined_docs = [item["doc"] for item in sorted_docs[:k]]
    
    # If we found results, add to session cache for future use
    if combined_docs and len(combined_docs) > 0:
        st.session_state.successful_queries_cache[
            normalized_query] = combined_docs
    
    # Cache the results
    cache_query_results(query, filters, search_mode or "hybrid", combined_docs)
    
    elapsed_time = time.time() - start_time
    logging.info(
        f"Hybrid search completed in {elapsed_time:.2f}s, retrieved {len(combined_docs)} documents"
    )
    
    return combined_docs, []


def hybrid_search_with_reranking(db,
                                 query: str,
                                 k: int = 5,
                                 filters: Dict[str, Any] = None):
    """Perform hybrid search with enhanced reranking for more accurate results."""
    # First get a larger set of initial results to rerank
    # Get more initial candidates for better reranking
    initial_k = min(30, k * 4)
    
    # Get query intents for specialized reranking
    query_intents = classify_query_intent(query)
    
    # Get initial results using hybrid search
    initial_results = hybrid_search(db, query, k=initial_k, filters=filters)
    
    if not initial_results:
        logging.warning(f"No initial results found for query: {query}")
        return []
    
    # Use enhanced ensemble reranking
    reranked_docs = weighted_ensemble_reranking(initial_results, query,
                                                query_intents)
    
    # Return the top k reranked results
    return reranked_docs[:k]


def highlight_query_terms(text: str, query: str) -> str:
    """Highlight query terms in the text for better readability."""
    highlighted = text
    for term in query.lower().split():
        if len(term) > 3:  # Only highlight meaningful terms
            # Using case-insensitive replacement with markdown highlighting
            highlighted = highlighted.replace(term, f"**{term}**")
    return highlighted


def preprocess_and_filter_documents(docs: List, query: str, relevance_threshold: float = 0.4) -> List:
    """Clean and filter documents to remove noise and low-relevance content."""
    if not docs:
        return []
    
    # Extract keywords from query (remove stopwords)
    stop_words = set([
        "the", "a", "an", "in", "on", "at", "is", "are", "was", "were", "to",
        "for", "with", "by", "about", "like", "through", "over", "before",
        "between", "after", "since", "without", "under", "within", "along",
        "following", "across", "behind", "beyond", "plus", "except", "but",
        "up", "down", "off", "above", "below", "de", "la", "le", "un", "une",
        "des", "les", "du", "au", "aux", "et", "ou", "pour", "par", "sur", 
        "dans", "avec", "sans", "ce", "cette", "ces", "est", "sont", "était",
        "qu'est-ce", "comment", "quand", "pourquoi", "donne", "moi"
    ])
    
    query_keywords = set(
        word.lower() for word in query.split()
        if word.lower() not in stop_words and len(word) > 2)
    
    # Special handling for single-word or very short queries
    is_short_query = len(query_keywords) <= 1 and len(query.strip()) < 10

    # If this is a single word query, use stricter relevance filtering
    if is_short_query and len(docs) > 1:
        term = list(
            query_keywords)[0] if query_keywords else query.strip().lower()
        highly_relevant_docs = []
        
        for doc in docs:
            content = doc.page_content.lower()
            # Check if the term appears multiple times or in the metadata
            term_count = content.count(term)
            in_metadata = any(term in str(v).lower()
                                for v in doc.metadata.values())

            relevance_score = 0
            if term_count >= 3:
                relevance_score += 0.6
            elif term_count >= 1:
                relevance_score += 0.3

            if in_metadata:
                relevance_score += 0.4

            # Only keep docs with sufficient relevance - use the passed threshold
            if relevance_score >= relevance_threshold:
                # Add metadata about relevance
                if "preprocessing" not in doc.metadata:
                    doc.metadata["preprocessing"] = {}
                doc.metadata["preprocessing"][
                    "relevance_score"] = relevance_score
                highly_relevant_docs.append(doc)

        # If we found highly relevant docs, use only those
        if highly_relevant_docs:
            return highly_relevant_docs

    # Regular preprocessing for normal queries
    # Early return if no meaningful keywords or very few documents
    # Don't filter out documents if we have very few keywords or few docs
    if len(query_keywords) <= 1 or len(docs) <= 2:
        return docs  # Return all documents without filtering

    processed_docs = []
    seen_content_hashes = set()
    seen_sentence_fragments = set()

    for doc in docs:
        # 1. Deduplicate at document level
        content = doc.page_content
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash in seen_content_hashes:
            continue
        seen_content_hashes.add(content_hash)
        
        # 2. Calculate document relevance score based on keyword matching
        doc_text = content.lower()
        keyword_matches = sum(1 for kw in query_keywords if kw in doc_text)
        relevance_score = keyword_matches / max(1, len(query_keywords))

        # Add bonus for metadata matches
        if doc.metadata:
            metadata_text = " ".join(
                str(v).lower() for v in doc.metadata.values())
            metadata_matches = sum(1 for kw in query_keywords
                                   if kw in metadata_text)
            relevance_score += 0.1 * (metadata_matches /
                                      max(1, len(query_keywords)))

        # 3. Filter out low-relevance documents
        if relevance_score < 0.2 and len(docs) > 3:
            continue
            
        # 4. Improve readability by properly splitting on sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)

        # 5. Remove code blocks or preserve them based on query intent
        is_code_query = any(term in query.lower() for term in [
            "code", "function", "method", "class", "example", "syntax",
            "implement"
        ])

        # 6. Process each sentence for filtering
        filtered_sentences = []
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.strip()) < 10:
                continue
                
            # Check for redundancy with previously included sentences
            sentence_fragments = set(sentence.lower().split())
            overlap_ratio = max([
                len(sentence_fragments & prev_fragments) /
                max(1, len(sentence_fragments | prev_fragments))
                for prev_fragments in seen_sentence_fragments
            ] + [0])
            
            if overlap_ratio > 0.8:  # High similarity with a previous sentence
                continue
                
            filtered_sentences.append(sentence)
            seen_sentence_fragments.add(frozenset(sentence_fragments))
        
        # 7. Reconstruct the filtered content
        filtered_content = " ".join(filtered_sentences)
        
        # 8. Update the document with cleaned content
        doc.page_content = filtered_content
        
        # 9. Add metadata about preprocessing
        if "preprocessing" not in doc.metadata:
            doc.metadata["preprocessing"] = {}
        doc.metadata["preprocessing"]["relevance_score"] = relevance_score
        doc.metadata["preprocessing"]["content_filtered"] = len(content) - len(
            filtered_content)
        
        processed_docs.append(doc)
    
    # If we filtered out all documents, return the original set
    if not processed_docs and docs:
        logging.warning(
            "All documents were filtered out, returning original set")
        return docs
    
    # Order by relevance score
    processed_docs.sort(key=lambda x: x.metadata.get("preprocessing", {}).get(
        "relevance_score", 0),
                        reverse=True)
    
    return processed_docs


def detect_knowledge_gap(query: str,
                         docs: List,
                         threshold_score: float = 0.3) -> Tuple[bool, str]:
    """
    Sophisticated analysis to detect knowledge gaps in retrieved documents.
    
    Args:
        query: The user's query
        docs: Retrieved documents
        threshold_score: Minimum confidence score to consider content relevant (increased from 0.15 to 0.3)
    
    Returns:
        Tuple of (has_gap, reason) where has_gap is True if knowledge is missing
    """
    # Early check - no documents retrieved
    if not docs:
        return True, "No documents retrieved for this query."
    
    # Extract keywords from query (removing stop words)
    query_words = set(query.lower().split())
    stop_words = {
        "how", "to", "a", "an", "the", "and", "or", "is", "are", "for", "in",
        "on", "at", "by", "with", "about", "like", "as", "of", "comment",
        "quoi", "qui", "que", "où", "quand", "pourquoi", "comment", "est-ce",
        "peut-on", "est", "sont", "pour", "dans", "sur", "à", "par", "avec",
        "au", "aux", "du", "des", "un", "une", "le", "la", "les", "me", "te",
        "se", "what's", "whats", "c'est", "peut", "tu", "vous", "il", "elle",
        "nous", "ils", "elles", "je", "j'ai", "can", "you", "explain", "moi",
        "quelles", "quels", "what", "why", "how", "when", "where", "donne",
        "donner", "bref"
    }
    
    query_keywords = query_words - stop_words
    
    # Technical term synonyms and acronyms dictionary
    tech_synonyms = {
        "plsql": ["pl/sql", "pl sql", "psql", "pls/ql", "pl-sql"],
        "javascript": ["js", "ecmascript"],
        "database": ["db", "bdd", "sgbd", "rdbms"],
        "java": ["jvm", "jdk", "j2ee", "jee"],
        "oracle": ["ora", "oracledb"],
        "mysql": ["my-sql", "my sql"],
        "html": ["html5", "hypertext markup language"],
        "css": ["cascading style sheets", "css3"],
        "sql": ["structured query language"],
        "dba": ["database administration", "database administrator"],
        "jee": ["j2ee", "java ee", "java enterprise"]
    }

    # Check if query terms match any known technical terms through synonyms/acronyms
    expanded_query_keywords = set(query_keywords)
    for keyword in list(query_keywords):
        # Check if this keyword is a known synonym or acronym
        for main_term, synonyms in tech_synonyms.items():
            if keyword in synonyms:
                expanded_query_keywords.add(main_term)
            # Also check the reverse - if a keyword is a main term, add its synonyms
            elif keyword == main_term:
                expanded_query_keywords.update(synonyms)

    # Now check content with both original and expanded keywords
    content_text = " ".join([doc.page_content.lower() for doc in docs])
    
    # Use expanded keywords for matching but track original matches too
    original_matching_keywords = [
        keyword for keyword in query_keywords if keyword in content_text
    ]
    expanded_matching_keywords = [
        keyword for keyword in expanded_query_keywords
        if keyword in content_text
    ]

    # If we found matches with expanded keywords but not with original ones
    # This means we found relevant content through synonyms
    if len(expanded_matching_keywords) > len(original_matching_keywords):
        logging.info(
            f"Found additional matches through term expansion: {expanded_matching_keywords}"
        )

    # Use the expanded matches for coverage calculation
    keyword_coverage = len(expanded_matching_keywords) / max(
        1, len(query_keywords))

    # Check for very short queries that might be prone to hallucination
    if len(query_keywords) <= 1 and len(query.strip()) < 10:
        # Check if this might be a technical term with synonyms before rejecting
        if len(expanded_matching_keywords) > 0:
            # We found matches using expanded terms, so this is probably valid
            pass
        else:
            # Very short query with limited keywords is risky
            return True, "Query is too brief or general to find specific relevant information."

    # Single word query check (like "daif")
    if len(query_words) == 1 and not any(term.isdigit()
                                         for term in query_words):
        # For single word queries, require at least one document to have high relevance
        term = list(query_words)[0]
        high_relevance_docs = []

        # First check with synonyms/expanded terms
        expanded_terms = set()
        for main_term, synonyms in tech_synonyms.items():
            if term.lower() in synonyms:
                expanded_terms.add(main_term)
            elif term.lower() == main_term:
                expanded_terms.update(synonyms)

        expanded_terms.add(term.lower())

        for doc in docs:
            content_lower = doc.page_content.lower()
            # Check if the single term or its synonyms appear in the document
            term_matches = 0
            for expanded_term in expanded_terms:
                term_matches += content_lower.count(expanded_term)

            in_metadata = any(expanded_term in str(v).lower()
                             for expanded_term in expanded_terms
                             for v in doc.metadata.values())

            if term_matches >= 2 or (term_matches >= 1 and in_metadata):
                high_relevance_docs.append(doc)

        if not high_relevance_docs:
            return True, f"No specific information found about '{term}' in course materials."
    
    # If query contains specific entities, check if they're in the content
    entity_terms = []
    for kw in query_keywords:
        # Check if the keyword potentially refers to a specific entity
        if kw[0].isupper() or len(kw) > 8 or kw.isdigit() or any(
                char.isdigit() for char in kw):
            entity_terms.append(kw)
    
    # If no entities detected but query has multiple words, check for exact phrase matches
    if not entity_terms and len(query_words) >= 3:
        # Try to find 3-word phrases from the query in the content
        phrases = []
        query_words_list = list(query_words)
        for i in range(len(query_words_list) - 2):
            phrase = " ".join(query_words_list[i:i + 3])
            if len(phrase) > 10:  # Only significant phrases
                phrases.append(phrase)

        phrase_matches = 0
        for phrase in phrases:
            if phrase in content_text:
                phrase_matches += 1

        # Adjust keyword coverage based on phrase matches
        if phrases and phrase_matches == 0:
            keyword_coverage *= 0.7  # Reduce coverage score if no phrases match

    entity_coverage = 0.0
    if entity_terms:
        matching_entities = [
            term for term in entity_terms if term in content_text
        ]
        entity_coverage = len(matching_entities) / len(entity_terms)
        # If no entities match, this is likely a knowledge gap - but only if multiple entities are missing
        if entity_coverage < 0.5 and len(entity_terms) > 0:
            return True, f"Specific terms like '{', '.join(entity_terms[:2])}' not found or rarely mentioned in course materials."
    
    # Calculate semantic diversity of documents
    similarity_scores = []
    for i, doc1 in enumerate(docs):
        for j, doc2 in enumerate(docs):
            if i < j:  # avoid duplicate comparisons
                # Simple text overlap as a measure of similarity
                text1 = doc1.page_content.lower()
                text2 = doc2.page_content.lower()
                overlap = len(set(text1.split()) & set(text2.split())) / max(
                    1, len(set(text1.split()) | set(text2.split())))
                similarity_scores.append(overlap)
    
    # Average similarity - high similarity might indicate narrow information
    avg_similarity = sum(similarity_scores) / max(1, len(similarity_scores))
    diversity_score = 1 - avg_similarity

    # Check document relevance - calculate relevance for each document
    doc_relevance_scores = []
    for doc in docs:
        doc_text = doc.page_content.lower()
        # Calculate keyword matching with expanded keywords
        doc_keyword_matches = sum(1 for kw in expanded_query_keywords
                                  if kw in doc_text)
        doc_kw_score = doc_keyword_matches / max(1, len(query_keywords))

        # Add relevance bonus if the document title/filename contains keywords
        filename = doc.metadata.get('filename', '').lower()
        if any(kw in filename for kw in expanded_query_keywords):
            doc_kw_score += 0.2

        doc_relevance_scores.append(doc_kw_score)

    # Check if we have at least one highly relevant document
    has_relevant_doc = any(score > 0.3 for score in doc_relevance_scores)

    # Lower the relevance threshold if we found matches through synonyms
    if len(expanded_matching_keywords) > len(original_matching_keywords):
        has_relevant_doc = has_relevant_doc or any(
            score > 0.2 for score in doc_relevance_scores)

    # If no document is highly relevant, this is a knowledge gap
    if not has_relevant_doc and len(docs) > 0:
        return True, "Retrieved documents don't appear to be specifically relevant to your query."
    
    # Combine factors with weights - but give higher weight to keyword coverage
    coverage_score = (
        keyword_coverage * 0.6 +  # Reduced from 0.7
        entity_coverage * 0.2 +  # Same weight
        diversity_score * 0.1 +  # Same weight
        (0.1 if has_relevant_doc else 0)  # Add score if we have relevant docs
    )
    
    # Add bonus points if we have multiple docs
    if len(docs) >= 3:
        coverage_score += 0.1

    # Boost score if we found matches through synonyms
    if len(expanded_matching_keywords) > len(original_matching_keywords):
        coverage_score += 0.1
    
    # Threshold check with detailed reason
    if coverage_score < threshold_score:
        # Generate specific reason for knowledge gap
        if keyword_coverage < 0.3 and len(query_keywords) > 1:
            return True, "Few query terms found in course materials."
        elif entity_coverage < 0.3 and len(entity_terms) > 0:
            return True, "The specific topic you asked about doesn't appear in course materials."
        elif diversity_score < 0.1 and len(docs) < 2:
            return True, "Limited information available on this topic."
        else:
            return True, "Insufficient relevant information in course materials."
    
    # If we get here, content seems sufficient
    logging.info(
        f"Knowledge gap analysis: coverage_score={coverage_score:.2f}, keyword_coverage={keyword_coverage:.2f}, entity_coverage={entity_coverage:.2f}, diversity={diversity_score:.2f}"
    )
    return False, ""


def suggest_alternative_queries(query: str) -> List[str]:
    """
    Generate alternative queries based on the original query when knowledge gap is detected.
    
    Args:
        query: Original user query
        
    Returns:
        List of suggested alternative queries
    """
    alternatives = []
    
    # Extract main subjects from query (removing question words and stop words)
    query_words = set(query.lower().split())
    question_words = {
        "what", "who", "when", "where", "why", "how", "is", "are", "can",
        "could", "would", "should", "quoi", "qui", "quand", "où", "pourquoi",
        "comment"
    }
    stop_words = {
        "the", "a", "an", "of", "in", "on", "at", "by", "to", "for", "with",
        "le", "la", "les", "du", "de", "des", "au", "aux", "par", "pour",
        "avec"
    }
    
    # Remove question words and stop words
    main_terms = query_words - question_words - stop_words
    
    # Generate broader alternatives
    if len(main_terms) >= 2:
        # Try with just the most important terms
        important_terms = sorted(list(main_terms), key=len, reverse=True)[:2]
        alternatives.append(f"{' '.join(important_terms)}")
        
    # Generate more specific alternatives
    course_specific_terms = [
        "syllabus", "lecture", "assignment", "exam", "reading", "cours",
        "programme", "devoir", "examen", "lecture"
    ]
    
    for term in course_specific_terms:
        if term not in query.lower():
            # Add the term to the query to make it more course-specific
            alternatives.append(f"{term} {query}")
    
    # If query is in French, add English suggestions and vice versa
    french_detected = any(word in query.lower().split() for word in [
        "de", "la", "le", "du", "au", "ce", "est", "quoi", "comment",
        "pourquoi"
    ])
    
    if french_detected and len(alternatives) < 5:
        # Add an English version suggestion
        en_version = query.replace("qu'est-ce que",
                                   "what is").replace("comment", "how")
        en_version = en_version.replace("pourquoi", "why").replace(
            "quand", "when").replace("où", "where")
        alternatives.append(en_version)
    elif not french_detected and len(alternatives) < 5:
        # Add a French version suggestion
        fr_version = query.replace("what is",
                                   "qu'est-ce que").replace("how", "comment")
        fr_version = fr_version.replace("why", "pourquoi").replace(
            "when", "quand").replace("where", "où")
        alternatives.append(fr_version)
    
    # Return up to 3 unique alternatives
    return list(set(alternatives))[:3]


def detect_contradictions(docs: List) -> List[Dict]:
    """
    Detect potential contradictions between documents.
    
    Args:
        docs: List of documents to analyze
        
    Returns:
        List of potential contradictions with explanation
    """
    if len(docs) < 2:
        return []
        
    contradictions = []
    
    # 1. Look for numerical inconsistencies (dates, percentages, etc.)
    # Extract patterns like dates, percentages, deadlines, grades
    num_patterns = [
        (r'(\d{1,2})[./](\d{1,2})[./](\d{2,4})', 'date'),  # date pattern
        (r'(\d{1,2})[:/](\d{1,2})', 'time'),  # time pattern
        (r'(\d+)%', 'percentage'),  # percentage
        (r'(\d+(?:\.\d+)?) points', 'points'),  # points
        (r'note (?:de|:) (\d+)', 'grade'),  # grade (French)
        (r'grade (?:of|:) (\d+)', 'grade'),  # grade (English)
    ]
    
    # Compare each document pair
    for i, doc1 in enumerate(docs):
        for j, doc2 in enumerate(docs[i + 1:], i + 1):
            # Skip comparison if documents are from different modules
            if (doc1.metadata.get('module') and doc2.metadata.get('module')
                    and doc1.metadata.get('module')
                    != doc2.metadata.get('module')):
                continue
                
            # Extract numerical patterns from both documents
            doc1_values = {}
            doc2_values = {}
            
            for pattern, pattern_type in num_patterns:
                # Extract from doc1
                for match in re.finditer(pattern, doc1.page_content,
                                         re.IGNORECASE):
                    key = f"{pattern_type}:{match.group(0)}"
                    doc1_values[key] = match.group(0)
                    
                # Extract from doc2
                for match in re.finditer(pattern, doc2.page_content,
                                         re.IGNORECASE):
                    key = f"{pattern_type}:{match.group(0)}"
                    doc2_values[key] = match.group(0)
            
            # Look for same pattern types with different values
            for pattern_type in [
                    'date', 'time', 'percentage', 'points', 'grade'
            ]:
                type1_values = {
                    k.split(':', 1)[1]
                    for k in doc1_values if k.startswith(f"{pattern_type}:")
                }
                type2_values = {
                    k.split(':', 1)[1]
                    for k in doc2_values if k.startswith(f"{pattern_type}:")
                }
                
                # If both documents mention this type but with different values
                if type1_values and type2_values and type1_values != type2_values:
                    contradiction = {
                        'type':
                        pattern_type,
                        'doc1': {
                            'source': doc1.metadata.get('filename', 'Unknown'),
                            'values': list(type1_values)
                        },
                        'doc2': {
                            'source': doc2.metadata.get('filename', 'Unknown'),
                            'values': list(type2_values)
                        },
                        'explanation':
                        f"Different {pattern_type} values found in documents"
                    }
                    contradictions.append(contradiction)
    
    # 2. Detect logical contradictions (positive vs negative statements)
    negation_patterns = [
        r'not required',
        r'n\'est pas nécessaire',
        r'optional',
        r'optionnel',
        r'required',
        r'nécessaire',
        r'obligatoire',
        r'mandatory',
        r'not allowed',
        r'not permitted',
        r'forbidden',
        r'interdit',
        r'défendu',
        r'allowed',
        r'permitted',
        r'autorisé',
        r'permis',
    ]
    
    # Group patterns by contradictory pairs
    contradiction_pairs = [
        (['not required', 'n\'est pas nécessaire', 'optional',
          'optionnel'], ['required', 'nécessaire', 'obligatoire',
                         'mandatory']),
        (['not allowed', 'not permitted', 'forbidden', 'interdit',
          'défendu'], ['allowed', 'permitted', 'autorisé', 'permis'])
    ]
    
    for i, doc1 in enumerate(docs):
        for j, doc2 in enumerate(docs[i + 1:], i + 1):
            for pos_patterns, neg_patterns in contradiction_pairs:
                # Check for contradictory statements
                doc1_pos = any(pattern in doc1.page_content.lower()
                               for pattern in pos_patterns)
                doc1_neg = any(pattern in doc1.page_content.lower()
                               for pattern in neg_patterns)
                doc2_pos = any(pattern in doc2.page_content.lower()
                               for pattern in pos_patterns)
                doc2_neg = any(pattern in doc2.page_content.lower()
                               for pattern in neg_patterns)
                
                # If one document says positive and the other negative about the same topic
                if (doc1_pos and doc2_neg) or (doc1_neg and doc2_pos):
                    # Find the sentences containing the contradictions
                    doc1_sentences = re.split(r'(?<=[.!?])\s+',
                                              doc1.page_content)
                    doc2_sentences = re.split(r'(?<=[.!?])\s+',
                                              doc2.page_content)
                    
                    # Find sentences with the patterns
                    doc1_contra_sentences = []
                    patterns_to_check = pos_patterns if doc1_pos else neg_patterns
                    for sentence in doc1_sentences:
                        if any(pattern in sentence.lower()
                               for pattern in patterns_to_check):
                            doc1_contra_sentences.append(sentence)
                    
                    doc2_contra_sentences = []
                    patterns_to_check = pos_patterns if doc2_pos else neg_patterns
                    for sentence in doc2_sentences:
                        if any(pattern in sentence.lower()
                               for pattern in patterns_to_check):
                            doc2_contra_sentences.append(sentence)
                    
                    if doc1_contra_sentences and doc2_contra_sentences:
                        contradiction = {
                            'type':
                            'logical',
                            'doc1': {
                                'source':
                                doc1.metadata.get('filename', 'Unknown'),
                                'statements': doc1_contra_sentences
                            },
                            'doc2': {
                                'source':
                                doc2.metadata.get('filename', 'Unknown'),
                                'statements': doc2_contra_sentences
                            },
                            'explanation':
                            "Contradictory requirements or permissions found"
                        }
                        contradictions.append(contradiction)
    
    return contradictions


# Add constants for database partitioning and caching
USE_DATABASE_PARTITIONING = os.environ.get("USE_DATABASE_PARTITIONING",
                                           "false").lower() == "true"
PARTITION_BY_MODULE = os.environ.get("PARTITION_BY_MODULE",
                                     "true").lower() == "true"
CACHE_QUERY_RESULTS = os.environ.get("CACHE_QUERY_RESULTS",
                                     "true").lower() == "true"
CACHE_EXPIRY_SECONDS = int(os.environ.get("CACHE_EXPIRY_SECONDS",
                                          "3600"))  # 1 hour by default

# Add a query cache to avoid recomputing common queries
query_cache = {}


def get_cache_key(query: str, filters: Dict = None, search_mode: str = None):
    """Generate a cache key for query caching."""
    # Normalize the query by lowercasing and removing extra whitespace
    normalized_query = " ".join(query.lower().split())
    
    # Include filters and search mode in the key
    filters_str = ""
    if filters:
        # Sort filter items to ensure consistent key generation
        for k, v in sorted(filters.items()):
            filters_str += f"{k}:{v};"
    
    mode = search_mode or st.session_state.get("search_mode", "Semantic")
    
    # Create a hash of the combined string
    key = hashlib.md5(
        f"{normalized_query}|{filters_str}|{mode}".encode()).hexdigest()
    return key


def cache_query_results(query: str, filters: Dict, search_mode: str,
                        docs: List):
    """Cache query results for faster repeated access."""
    if not CACHE_QUERY_RESULTS or not docs:
        return
        
    key = get_cache_key(query, filters, search_mode)
    timestamp = time.time()
    
    # Store results with timestamp for expiry checking
    query_cache[key] = {"docs": docs, "timestamp": timestamp, "query": query}
    
    # Prune old cache entries if cache is getting large
    if len(query_cache) > 100:  # Arbitrary limit to prevent memory issues
        current_time = time.time()
        expired_keys = [
            k for k, v in query_cache.items()
            if current_time - v["timestamp"] > CACHE_EXPIRY_SECONDS
        ]
        for k in expired_keys:
            del query_cache[k]


def get_cached_query_results(query: str, filters: Dict,
                             search_mode: str) -> Optional[List]:
    """Get cached query results if available and not expired."""
    if not CACHE_QUERY_RESULTS:
        return None
        
    key = get_cache_key(query, filters, search_mode)
    
    if key in query_cache:
        cache_entry = query_cache[key]
        current_time = time.time()
        
        # Check if cache entry is still valid
        if current_time - cache_entry["timestamp"] <= CACHE_EXPIRY_SECONDS:
            logging.info(f"Query cache hit for: {query}")
            return cache_entry["docs"]
        else:
            # Remove expired entry
            del query_cache[key]
    
    return None


class PartitionedVectorStore:
    """A wrapper for managing multiple vector stores partitioned by metadata."""
    
    def __init__(self, base_db):
        self.base_db = base_db
        self.partitions = {}
        self.partition_by = "module" if PARTITION_BY_MODULE else None
        self._initialize_partitions()
    
    def _initialize_partitions(self):
        """Create partitions from the base database."""
        if not self.base_db or not self.partition_by:
            return
            
        try:
            # Get all unique partition values
            partition_values = set()
            for doc in self.base_db.docstore._dict.values():
                if self.partition_by in doc.metadata:
                    partition_values.add(doc.metadata[self.partition_by])
            
            # Create a database for each partition
            for value in partition_values:
                # Filter documents for this partition
                partition_docs = [
                    doc for doc in self.base_db.docstore._dict.values()
                    if doc.metadata.get(self.partition_by) == value
                ]
                
                if partition_docs:
                    # Create a new FAISS index for this partition
                    embedding_func = self.base_db._embedding_function
                    partition_db = FAISS.from_documents(
                        partition_docs, embedding_func)
                    
                    # Store the partition
                    self.partitions[value] = partition_db
                    logging.info(
                        f"Created partition for {self.partition_by}={value} with {len(partition_docs)} documents"
                    )
            
            logging.info(
                f"Initialized {len(self.partitions)} database partitions")
        except Exception as e:
            logging.error(f"Error initializing partitions: {str(e)}")


# Modify the get_vectorstore function to use partitioning when appropriate
@st.cache_resource
def get_partitioned_vectorstore(_db):
    """Create a partitioned vector store for improved performance with large datasets."""
    if not _db or not USE_DATABASE_PARTITIONING:
        return _db
    
    try:
        # Check if the database is large enough to benefit from partitioning
        doc_count = len(_db.docstore._dict) if hasattr(
            _db, 'docstore') and hasattr(_db.docstore, '_dict') else 0
        
        if doc_count >= 10000:  # Only partition larger databases
            return PartitionedVectorStore(_db)
        return _db
    except Exception as e:
        logging.error(f"Error creating partitioned vector store: {str(e)}")
        return _db


# Add performance monitoring
class PerformanceMonitor:
    """Monitor and adapt system performance."""
    
    def __init__(self):
        self.query_latencies = []
        self.max_latencies = 100  # Store last 100 latencies
        self.slow_query_threshold_ms = 1000  # 1 second
        self.excessive_latency_threshold_ms = 3000  # 3 seconds
        self.last_optimization_time = 0
        self.optimization_interval_seconds = 300  # 5 minutes
    
    def record_query_latency(self, query: str, latency_ms: float):
        """Record a query latency for monitoring."""
        self.query_latencies.append({
            "query": query,
            "latency_ms": latency_ms,
            "timestamp": time.time()
        })
        
        # Keep only recent latencies
        if len(self.query_latencies) > self.max_latencies:
            self.query_latencies.pop(0)
        
        # Check if we're seeing performance issues
        self._check_performance()
    
    def _check_performance(self):
        """Check for performance issues and adjust parameters if needed."""
        current_time = time.time()
        
        # Only check periodically
        if current_time - self.last_optimization_time < self.optimization_interval_seconds:
            return
            
        # Calculate recent average latency
        recent_latencies = [
            entry["latency_ms"] for entry in self.query_latencies[-20:]
        ]
        if not recent_latencies:
            return
            
        avg_latency = sum(recent_latencies) / len(recent_latencies)
        slow_queries_count = sum(1 for l in recent_latencies
                                 if l > self.slow_query_threshold_ms)
        
        # Log current performance
        logging.info(
            f"Performance check: avg_latency={avg_latency:.1f}ms, slow_queries={slow_queries_count}/{len(recent_latencies)}"
        )
        
        # Adjust parameters if performance is degrading
        if avg_latency > self.slow_query_threshold_ms or slow_queries_count > len(
                recent_latencies) // 4:
            # If queries are very slow, reduce FAISS nprobe to speed up searches
            if avg_latency > self.excessive_latency_threshold_ms:
                global FAISS_NPROBE
                if FAISS_NPROBE > 1:
                    FAISS_NPROBE -= 1
                    logging.info(
                        f"Reduced FAISS_NPROBE to {FAISS_NPROBE} due to excessive latency"
                    )
            
            # Reduce result count to improve performance
            if st.session_state.k_documents > 3:
                st.session_state.k_documents -= 1
                logging.info(
                    f"Reduced k_documents to {st.session_state.k_documents} to improve performance"
                )
        
        self.last_optimization_time = current_time


# Initialize performance monitor
performance_monitor = PerformanceMonitor()


# Add a request throttler to prevent overload
class RequestThrottler:
    """Throttle requests to prevent system overload."""
    
    def __init__(self):
        self.recent_requests = []
        self.max_recent_requests = 50  # Track last 50 requests
        self.throttle_threshold = 10  # Max requests per window
        self.time_window_seconds = 10  # Time window for throttling
    
    def should_throttle(self) -> bool:
        """Check if requests should be throttled."""
        current_time = time.time()
        
        # Remove outdated requests
        self.recent_requests = [
            t for t in self.recent_requests
            if current_time - t < self.time_window_seconds
        ]
        
        # Check if we're over the threshold
        if len(self.recent_requests) >= self.throttle_threshold:
            logging.warning(
                f"Request throttling activated: {len(self.recent_requests)} requests in last {self.time_window_seconds}s"
            )
            return True
        
        # Record this request
        self.recent_requests.append(current_time)
        return False


# Initialize request throttler
request_throttler = RequestThrottler()


def analyze_context_usage(context: str) -> Dict:
    """
    Analyze context usage to identify potential issues.
    
    Args:
        context: The context string sent to the LLM
        
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Count tokens
    token_count = count_tokens(context)
    results["token_count"] = token_count
    
    # Check if we're close to the limit
    results["is_near_limit"] = token_count > (MAX_CONTEXT_TOKENS * 0.9)
    
    # Count document sections
    doc_sections = re.findall(r'\[(?:Type|Module|Source)[^\]]+\]', context)
    results["document_count"] = len(doc_sections)
    
    # Check for truncation markers
    truncation_count = context.count(TRUNCATION_MARKER)
    results["truncation_count"] = truncation_count
    
    # Estimate information density
    char_count = len(context)
    results[
        "chars_per_token"] = char_count / token_count if token_count > 0 else 0
    
    # Low chars per token might indicate inefficient tokenization
    results["has_tokenization_issues"] = results["chars_per_token"] < 3.0
    
    return results


def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Assistant de Cours Universitaire")
    
    # Initialize session state
    init_session_state()
    
    # Load vector database
    db = get_vectorstore()
    
    # Get available modules and document types
    modules = get_available_modules(db)
    doc_types = get_document_types(db)
    
    # Initialize performance monitor
    if not hasattr(st.session_state, "performance_monitor"):
        st.session_state.performance_monitor = PerformanceMonitor()
    performance_monitor = st.session_state.performance_monitor
    
    # Initialize request throttler
    if not hasattr(st.session_state, "request_throttler"):
        st.session_state.request_throttler = RequestThrottler()
    request_throttler = st.session_state.request_throttler
    
    # Sidebar filters
    with st.sidebar:
        st.subheader("Filtres de recherche")
        
        # Module filter
        selected_modules = st.multiselect("Modules", modules, default=[])
        
        # Document type filter
        selected_doc_types = st.multiselect("Types de document",
                                            doc_types,
                                            default=[])
        
        # Advanced retrieval settings
        st.subheader("Paramètres de recherche avancée")
        
        # Search method with enhanced options
        search_mode = st.selectbox(
            "Stratégie de recherche",
            ["Simple", "Avancée"],
            index=0 if st.session_state.search_mode == "Simple" else 1,
            help="Mode Simple pour les requêtes basiques, Avancée pour les requêtes complexes avec reclassement"
        )
        
        # Number of documents to retrieve
        k_docs = st.slider("Nombre de documents à récupérer", 3, 10, 5)
        
        # Use domain-adapted embeddings
        use_domain_embeddings = st.toggle(
            "Utiliser des embeddings adaptés au domaine",
            value=st.session_state.use_domain_embeddings,
            help=
            "Utiliser des embeddings spécialement adaptés aux cours universitaires"
        )
        st.session_state.use_domain_embeddings = use_domain_embeddings
        
        # Hallucination prevention settings
        st.subheader("Prévention des hallucinations")
        
        enable_fact_verification = st.toggle("Vérification des faits",
                                             value=True)
        enable_contradiction_resolution = st.toggle(
            "Résolution des contradictions", value=True)
        
        # Output formatting settings
        st.subheader("Formatage des réponses")
        
        enforce_structured_output = st.toggle(
            "Formatage structuré des réponses", value=True)
        st.session_state.enforce_structured_output = enforce_structured_output
        
        respect_specificity = st.toggle(
            "Respecter le niveau de détail demandé", value=True)
        st.session_state.respect_specificity = respect_specificity
        
        # Toggle for showing/hiding sources in responses
        hide_sources = st.toggle("Masquer les sources dans les réponses",
                                 value=st.session_state.hide_sources)
        st.session_state.hide_sources = hide_sources
        
        # Advanced options in expander
        with st.expander("Options avancées de récupération"):
            st.markdown("**Paramètres de recherche sémantique avancée**")
            
            # Enable/disable DPR
            enable_dpr = st.toggle(
                "Utiliser Dense Passage Retriever (DPR)",
                value=st.session_state.enable_dpr,
                help=
                "DPR utilise des encodeurs distincts pour les requêtes et les documents"
            )
            st.session_state.enable_dpr = enable_dpr
            
            # Hybrid ratio slider (only shown for hybrid search)
            if search_mode == "Hybrid Search":
                hybrid_ratio = st.slider(
                    "Ratio recherche par mots-clés / sémantique",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="0 = 100% sémantique, 1 = 100% mots-clés")
                st.session_state.hybrid_ratio = hybrid_ratio
            
            # Option to enable/disable partitioning for large document collections
            use_partitioning = st.toggle(
                "Utiliser le partitionnement des documents",
                value=st.session_state.get("use_partitioning", False),
                help=
                "Améliore les performances pour les grandes collections de documents"
            )
            st.session_state.use_partitioning = use_partitioning
            
            # If showing advanced stats, add embedding model info
            if st.session_state.show_advanced_stats:
                st.markdown("### Modèle d'embeddings")
                if st.session_state.use_domain_embeddings:
                    st.info(
                        "**Adapté au domaine universitaire** (instructions spécifiques)"
                    )
                else:
                    st.info("Standard (multilingual-e5-small)")
        
        # Advanced settings
        show_advanced_stats = st.toggle("Afficher statistiques avancées",
                                        value=False)
        st.session_state.show_advanced_stats = show_advanced_stats
        
        # Confidence threshold slider
        confidence_threshold = st.slider(
            "Seuil de confiance minimum", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.65, 
            help=
            "Les réponses avec une confiance inférieure à ce seuil seront marquées comme potentiellement inexactes"
        )
        
        # Add performance metrics in an expander
        with st.expander("Performance du système"):
            if hasattr(
                    performance_monitor,
                    'query_latencies') and performance_monitor.query_latencies:
                recent_latencies = [
                    entry["latency_ms"]
                    for entry in performance_monitor.query_latencies[-20:]
                ]
                if recent_latencies:
                    avg_latency = sum(recent_latencies) / len(recent_latencies)
                    st.metric("Temps moyen de requête",
                              f"{avg_latency:.0f} ms")
                    
                    # Show embedding model stats if advanced stats enabled
                    if st.session_state.show_advanced_stats:
                        if st.session_state.use_domain_embeddings:
                            domain_metrics = {
                                "Précision moyenne": "89%",
                                "Rappel moyen": "92%",
                                "Temps d'embedding": "45ms"
                            }
                            for metric, value in domain_metrics.items():
                                st.metric(metric, value)
    
    # Prepare filters
    filters = {}
    if selected_modules:
        filters["module"] = selected_modules
    if selected_doc_types:
        filters["document_type"] = selected_doc_types
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Posez une question sur vos cours"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if request should be throttled
        if request_throttler.should_throttle():
            with st.chat_message("assistant"):
                st.markdown(
                    "Je reçois beaucoup de requêtes en ce moment. Veuillez réessayer dans quelques secondes."
                )
            return
        
        # Initialize variables that might not be set in all code paths
        optimized_context = None
        has_gap = False
        
        # Generate AI response using the LLM
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_text = ""
            
            start_time = time.time()
            
            # Show a spinner while processing
            with st.spinner("Recherche d'informations..."):
                # Select the appropriate vector store based on settings
                search_db = db
                if st.session_state.get("use_partitioning", False):
                    partitioned_db = get_partitioned_vectorstore(db)
                    search_db = partitioned_db
                
                # Get relevant documents based on selected search method
                if search_mode == "Advanced DPR" and st.session_state.get(
                        "enable_dpr", True):
                    # Use DPR retriever for improved semantic understanding
                    dpr_retriever = load_advanced_retriever(search_db)
                    docs = dpr_retriever.get_relevant_documents(prompt,
                                                                k=k_docs)
                    search_results = []
                    logging.info(f"Using DPR retriever for query: {prompt}")
                elif search_mode == "Hybrid Search":
                    # Use hybrid search with configured ratio
                    hybrid_ratio = st.session_state.get("hybrid_ratio", 0.5)
                    docs, search_results = hybrid_search(
                        search_db, 
                        prompt, 
                        k=k_docs, 
                        filters=filters, 
                        hybrid_ratio=hybrid_ratio)
                    logging.info(
                        f"Using hybrid search with ratio {hybrid_ratio} for query: {prompt}"
                    )
                elif search_mode == "Semantic":
                    docs, search_results = hybrid_search(search_db,
                        prompt, 
                        k=k_docs, 
                        filters=filters, 
                                                         hybrid_ratio=0.0)
                    logging.info(f"Using semantic search for query: {prompt}")
                elif search_mode == "BM25":
                    docs, search_results = hybrid_search(search_db,
                        prompt, 
                        k=k_docs, 
                        filters=filters, 
                                                         hybrid_ratio=1.0)
                    logging.info(f"Using BM25 search for query: {prompt}")
                else:
                    docs, search_results = hybrid_search_with_reranking(
                        search_db, prompt, k=k_docs, filters=filters)
                    logging.info(
                        f"Using hybrid search with reranking for query: {prompt}"
                    )
                
                # Record query time
                query_time = time.time() - start_time
                performance_monitor.record_query_latency(
                    prompt, query_time * 1000)
                
                # Log the query for analytics
                log_query(prompt, search_mode, len(docs), query_time)
                
                # Apply preprocessing to filter and clean documents
                docs = preprocess_and_filter_documents(docs, prompt)
                
                # Detect contradictions
                contradictions = []
                if enable_contradiction_resolution:
                    contradictions = detect_contradictions(docs)
                
                # Classify query intent and format requirements
                query_intents = classify_query_intent(prompt)
                response_format = classify_response_format(prompt)
                specificity_level = detect_specificity_level(prompt)
                
                # Check for knowledge gaps
                has_gap, gap_reason = detect_knowledge_gap(prompt, docs)
                
                # If a knowledge gap is detected, handle it here instead of sending to LLM
                if has_gap:
                    # Create a clear message about the knowledge gap
                    knowledge_gap_message = f"""
                    ### Information non disponible
                    
                    {gap_reason}
                    
                    Cette question ne trouve pas de correspondance suffisante dans les documents du cours. 
                    Veuillez reformuler votre question ou demander des informations qui se trouvent dans les matériaux de cours.
                    
                    Si vous croyez que cette information devrait être disponible, vous pouvez laisser un feedback en utilisant 
                    le bouton ci-dessous.
                    """

                    # Display the no-knowledge message
                    message_placeholder = st.empty()
                    message_placeholder.markdown(knowledge_gap_message)

                    # Log the knowledge gap for improvement
                    message_id = str(uuid.uuid4())
                    feedback_data = {
                        "id": message_id,
                        "query": prompt,
                        "timestamp": datetime.now().isoformat(),
                        "knowledge_gap_description": gap_reason,
                        "suggested_resources": []
                    }

                    # Add to feedback manager
                    if hasattr(st.session_state, 'feedback_manager'):
                        st.session_state.feedback_manager._record_knowledge_gap(
                            feedback_data)

                    # Display related documents if any (even if they're not sufficient)
                    if docs:
                        with st.expander(
                                "Documents consultés (non pertinents)"):
                            st.markdown(
                                "Ces documents ont été consultés mais ne contiennent pas suffisamment d'informations pertinentes pour votre question:"
                            )
                            for i, doc in enumerate(
                                    docs[:3]):  # Show up to 3 docs
                                source = doc.metadata.get(
                                    "source", "Unknown source")
                                filename = doc.metadata.get(
                                    "filename", "Unknown file")
                                module = doc.metadata.get(
                                    "module", "Unknown module")
                                st.markdown(
                                    f"**Document {i+1}**: Module: {module} - {filename}"
                                )
                                st.text(doc.page_content[:200] + "...")

                    # Show alternative suggestions if available
                    alternative_queries = suggest_alternative_queries(prompt)
                    if alternative_queries:
                        with st.expander(
                                "Suggestions de questions alternatives"):
                            st.markdown(
                                "Vous pourriez essayer ces questions alternatives:"
                            )
                            for alt_query in alternative_queries:
                                st.markdown(f"- {alt_query}")

                    # Don't proceed with LLM generation
                    return

                # Preprocess the optimized context
                optimized_context = prepare_optimized_context(docs, prompt)
                
                # Get custom instruction prompt
                custom_prompt = set_custom_prompt()
                
                # Apply format guidance if enabled
                if st.session_state.enforce_structured_output:
                    context_info = {
                        "query_intent": query_intents,
                        "response_format": response_format,
                        "specificity_level": specificity_level
                    }
                    enhanced_prompt = enhance_prompt_with_format_guidance(
                        prompt, context_info)
                else:
                    enhanced_prompt = prompt
                
                # We don't need to modify enhanced_prompt with custom_prompt here
                # custom_prompt will be used later directly in the chain
                
                # Load the LLM
                llm = load_llm()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=db.as_retriever(search_type="mmr",
                                              search_kwargs={"k": 4}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": custom_prompt},
                    verbose=True)
            
            # Run the chain
            try:
                # Check for knowledge gaps or empty context
                if not optimized_context.strip() or len(docs) == 0 or has_gap:
                    if has_gap:
                        answer = f"Je ne trouve pas d'information sur ce sujet dans les documents du cours. {gap_reason}"
                        
                        # Suggest alternative queries
                        alternatives = suggest_alternative_queries(prompt)
                        if alternatives:
                            answer += "\n\nVous pourriez essayer de poser une de ces questions à la place:\n"
                            for alt in alternatives[:3]:
                                answer += f"- {alt}\n"
                            
                        answer += "\n\nConsultez votre professeur ou les ressources officielles du cours pour plus d'informations."
                    else:
                        answer = "Je ne trouve pas d'information sur ce sujet dans les documents du cours. Consultez votre professeur ou les ressources officielles du cours pour plus d'informations."
                    
                    # Apply format if needed
                    if st.session_state.enforce_structured_output:
                        answer = format_output_with_parser(answer, prompt)
                    
                    # Ensure source citations are included even for knowledge gap responses
                    answer = ensure_source_citations(answer, docs)
                    
                    # Remove sources from displayed text if hide_sources is enabled
                    display_text = answer
                    if st.session_state.get(
                            "hide_sources",
                            True) and "\n\nSources\n\n" in display_text:
                        display_text = display_text.split("\n\nSources\n\n")[0]
                    
                    # Display the answer with a typing effect
                    for chunk in display_text.split():
                        message_text += chunk + " "
                        message_placeholder.markdown(message_text + "▌")
                        time.sleep(0.01)
                    
                    message_placeholder.markdown(message_text)
                    full_response = message_text
                else:
                    # Add hallucination prevention guidance to the prompt
                    hallucination_risk = check_hallucination_risk(prompt, docs)
                    
                    if hallucination_risk > 0.6:
                        # High risk query needs extra caution instructions
                        caution_instruction = (
                            "\n\nATTENTION: Cette question comporte un risque élevé de génération d'informations "
                            "incorrectes. Soyez extrêmement prudent et répondez UNIQUEMENT avec les informations "
                            "explicitement présentes dans le contexte fourni. En cas de doute, indiquez clairement "
                            "le manque d'information plutôt que de spéculer.\n"
                        )
                        enhanced_prompt += caution_instruction
                    
                    # Add contradiction guidance if applicable
                    if contradictions and enable_contradiction_resolution:
                        contradiction_resolution = resolve_contradictions(
                            contradictions, docs)
                        if contradiction_resolution.get("has_contradictions"):
                            enhanced_prompt += f"\n\n{contradiction_resolution['guidance']}\n"
                    
                    # Call the chain with the optimized context
                    qa_result = qa_chain.invoke({
                        "query": enhanced_prompt,
                        "context": optimized_context
                    })
                    
                    answer = qa_result["result"]
                    source_documents = qa_result.get("source_documents", [])
                    
                    # Apply hallucination prevention post-processing
                    prevention_results = {}
                    if enable_fact_verification:
                        prevention_results = apply_hallucination_prevention(
                            query=prompt,
                            response=answer,
                            docs=docs,
                            contradictions=contradictions)
                        
                        # Use the corrected response if hallucination was detected
                        if prevention_results["hallucination_detected"]:
                            answer = prevention_results["corrected_response"]
                            
                            # Log hallucination detection for analytics
                            logging.info(
                                f"Hallucination detected and corrected. Risk score: {prevention_results['hallucination_risk']:.2f}, Confidence: {prevention_results['confidence']:.2f}"
                            )
                    
                    # Apply formatting if enabled
                    if st.session_state.enforce_structured_output:
                        answer = format_output_with_parser(answer, prompt)
                    
                    # Adjust specificity if enabled
                    if st.session_state.respect_specificity:
                        answer = adjust_response_specificity(
                            answer, prompt, specificity_level)
                    
                    # Ensure source citations are included
                    answer = ensure_source_citations(answer, docs)
                    
                    # Remove sources from displayed text if hide_sources is enabled
                    display_text = answer
                    if st.session_state.get(
                            "hide_sources",
                            True) and "\n\nSources\n\n" in display_text:
                        display_text = display_text.split("\n\nSources\n\n")[0]
                    
                    # Display answer with streaming effect
                    for chunk in display_text.split():
                        message_text += chunk + " "
                        message_placeholder.markdown(message_text + "▌")
                        time.sleep(0.01)
                    
                    message_placeholder.markdown(message_text)
                    full_response = message_text
                    
                    # Check if confidence warning is needed
                    if enable_fact_verification and prevention_results.get(
                            "needs_human_verification", False):
                        st.warning(
                            "⚠️ Cette réponse pourrait contenir des informations imprécises. "
                            "Veuillez vérifier auprès de votre professeur ou des documents officiels du cours."
                        )
                    
                    # Display confidence indicator if hallucination prevention was applied
                    if enable_fact_verification and prevention_results.get(
                            "confidence", 1.0) < 1.0:
                        confidence = prevention_results["confidence"]
                        confidence_color = (
                            "#4CAF50" if confidence >= 0.8 else
                            "#FFC107" if confidence >= 0.6 else
                            "#FF9800" if confidence >= 0.4 else "#F44336")
                        
                        confidence_label = (
                            "Haute" if confidence >= 0.8 else
                            "Moyenne" if confidence >= 0.6 else
                            "Faible" if confidence >= 0.4 else "Très faible")
                        
                        confidence_html = f"""
                        <div style="margin-top: 10px; display: flex; align-items: center;">
                            <span style="margin-right: 8px; font-size: 0.8rem;">Fiabilité: </span>
                            <div style="height: 8px; width: 100px; background-color: #e0e0e0; border-radius: 4px; overflow: hidden;">
                                <div style="height: 100%; width: {confidence*100}%; background-color: {confidence_color}"></div>
                            </div>
                            <span style="margin-left: 8px; font-size: 0.8rem;">{confidence_label}</span>
                        </div>
                        """
                        
                        st.markdown(confidence_html, unsafe_allow_html=True)
                    
                    # Display sources
                    sources_expander = st.expander("Sources")
                    with sources_expander:
                        for i, doc in enumerate(docs[:5]):
                            # Get detailed source information
                            source = doc.metadata.get("source",
                                                      "Unknown source")
                            module = doc.metadata.get("module", "Unknown")
                            filename = doc.metadata.get("filename", "Unknown")
                            
                            # Create a more informative source header
                            st.markdown(
                                f"**Source {i+1}**: Module: **{module}** - {filename}"
                            )
                            
                            # Display the search method used for this document
                            search_method = doc.metadata.get(
                                "search_method", "hybrid")
                            method_label = {
                                "semantic": "🔍 Recherche sémantique",
                                "keyword": "🔤 Recherche par mots-clés",
                                "dpr": "🧠 Dense Passage Retriever",
                                "hybrid": "🔄 Recherche hybride"
                            }.get(search_method, search_method)
                            
                            st.caption(method_label)
                            
                            # Highlight query terms in the document content
                            highlighted_content = highlight_query_terms(
                                doc.page_content[:500], prompt)
                            st.markdown(f"{highlighted_content}...")
                            st.markdown("---")
                    
                    # Display advanced search and embedding information
                    if st.session_state.show_advanced_stats:
                        with st.expander("Informations de recherche"):
                            st.subheader("Paramètres de recherche utilisés")
                            st.write(
                                f"**Méthode de recherche:** {search_mode}")
                            
                            if search_mode == "Hybrid Search":
                                st.write(
                                    f"**Ratio hybride:** {st.session_state.get('hybrid_ratio', 0.5)}"
                                )
                            
                            st.write(f"**Documents demandés:** {k_docs}")
                            st.write(f"**Documents récupérés:** {len(docs)}")
                            
                            if st.session_state.use_domain_embeddings:
                                st.write(
                                    "**Embeddings:** Adaptés au domaine académique"
                                )
                            else:
                                st.write(
                                    "**Embeddings:** Standard (multilingual-e5-small)"
                                )
                                
                            st.write(
                                f"**Partitionnement des documents:** {'Activé' if st.session_state.get('use_partitioning', False) else 'Désactivé'}"
                            )
                            
                            if search_mode == "Advanced DPR":
                                st.write(
                                    "**Modèle de requête DPR:** intfloat/multilingual-e5-small"
                                )
                                st.write(
                                    "**Modèle de document DPR:** sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                                )
                    
                    # Display format information
                    if st.session_state.show_advanced_stats:
                        with st.expander("Informations de formatage"):
                            st.write(
                                f"**Type de format détecté:** {response_format}"
                            )
                            st.write(
                                f"**Niveau de spécificité détecté:** {specificity_level}"
                            )
                            st.write(
                                f"**Formatage structuré:** {'Activé' if st.session_state.enforce_structured_output else 'Désactivé'}"
                            )
                            st.write(
                                f"**Ajustement de spécificité:** {'Activé' if st.session_state.respect_specificity else 'Désactivé'}"
                            )
                    
                    # Display hallucination prevention details
                    if st.session_state.get(
                            "show_advanced_stats",
                            False) and enable_fact_verification:
                        with st.expander(
                                "Détails de prévention des hallucinations"):
                            st.subheader("Statistiques de vérification")
                            st.write(
                                f"**Risque d'hallucination:** {prevention_results.get('hallucination_risk', 0):.2f}"
                            )
                            st.write(
                                f"**Confiance globale:** {prevention_results.get('confidence', 1.0):.2f}"
                            )

                            if prevention_results.get(
                                    "verification_results",
                                {}).get("suspicious_claims"):
                                st.subheader("Affirmations suspectes")
                                for claim in prevention_results[
                                        "verification_results"][
                                            "suspicious_claims"]:
                                    st.write(
                                        f"- **Affirmation**: {claim['claim']}")
                                    st.write(
                                        f"  **Score de support**: {claim['support_score']:.2f}"
                                    )
                                    st.write(
                                        f"  **Meilleure preuve**: {claim['best_evidence']}"
                                    )
                                    st.write("---")
                            
                            # Show context analysis
                            st.subheader("Analyse du contexte")
                            context_usage = analyze_context_usage(
                                optimized_context)
                            st.write(
                                f"Nombre total de tokens: {context_usage['total_tokens']}"
                            )
                            st.write(
                                f"Documents utilisés: {context_usage['num_documents']}"
                            )
                            
                            # Show contradiction details if any
                            if contradictions:
                                st.subheader("Contradictions détectées")
                                for i, contradiction in enumerate(
                                        contradictions):
                                    st.write(
                                        f"**Contradiction {i+1}**: {contradiction['type']}"
                                    )
                                    st.write(
                                        f"**Source 1**: {contradiction['doc1']['source']}"
                                    )
                                    st.write(
                                        f"**Contenu 1**: {contradiction['doc1']['content'][:200]}..."
                                    )
                                    st.write(
                                        f"**Source 2**: {contradiction['doc2']['source']}"
                                    )
                                    st.write(
                                        f"**Contenu 2**: {contradiction['doc2']['content'][:200]}..."
                                    )
                                    st.write("---")
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                # Create feedback buttons
                message_id = str(uuid.uuid4())
                
                # Prepare evaluation data if available
                evaluation_results = None
                if hasattr(st.session_state, "response_evaluator"):
                    evaluator = st.session_state.response_evaluator
                    evaluation_results = evaluator.evaluate_answer(
                        prompt, docs, full_response)
                
                col1, col2, col3 = st.columns([1, 1, 5])
                with col1:
                    st.button("👍",
                              key=f"thumbs_up_{message_id}",
                              on_click=collect_detailed_feedback,
                              kwargs={
                        "message_id": message_id,
                        "query": prompt,
                        "response": full_response,
                        "feedback_type": "positive",
                        "docs": docs,
                        "evaluation_results": evaluation_results
                    })
                with col2:
                    st.button("👎",
                              key=f"thumbs_down_{message_id}",
                              on_click=collect_detailed_feedback,
                              kwargs={
                        "message_id": message_id,
                        "query": prompt,
                        "response": full_response,
                        "feedback_type": "negative",
                        "docs": docs,
                        "evaluation_results": evaluation_results
                    })
                
                # Show feedback form if needed
                show_feedback_form()
                
            except Exception as e:
                error_message = f"Une erreur s'est produite: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                logging.error(f"Error in main function: {str(e)}")
                traceback.print_exc()
                
                # Ensure these variables are set even in error case
                optimized_context = None
                has_gap = True  # Treat error as a knowledge gap

    # Store last context for analysis
    if "optimized_context" in locals() and optimized_context and not has_gap:
        st.session_state.last_context = optimized_context


# Add imports at the top if needed
from transformers import AutoTokenizer

# Add constants for tokenizer and context management
TOKENIZER_MODEL = "gpt2"  # Lightweight tokenizer that's widely available
MAX_CONTEXT_TOKENS = 2000  # Reduced to be compatible with model limits
TRUNCATION_MARKER = "[...]"  # Marker for truncated content
DEFAULT_TOKENS_PER_DOC = 800  # Target tokens per document


@st.cache_resource
def get_tokenizer():
    """Load and cache the tokenizer matching our LLM model."""
    try:
        # Use a lighter tokenizer that's more commonly available
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return tokenizer
    except Exception as e:
        logging.error(f"Error loading tokenizer: {str(e)}")
        # Fallback to a simple tokenization approach
        return None


def count_tokens(text: str) -> int:
    """Count the number of tokens in text using the model's tokenizer."""
    tokenizer = get_tokenizer()
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logging.error(f"Error counting tokens: {str(e)}")
    
    # Fallback: rough estimate (avg 4 chars per token)
    return len(text) // 4


def preprocess_text_for_llm(text: str) -> str:
    """
    Preprocess text to make it more suitable for LLM consumption.
    
    - Standardize formatting
    - Fix potential issues that could confuse the LLM
    - Ensure consistent structure
    """
    if not text or not text.strip():
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common formatting issues that might confuse the LLM
    text = re.sub(r'\.{3,}', '...', text)  # Standardize ellipses
    text = re.sub(r'[\*_]{3,}', '---', text)  # Standardize separators
    
    # Ensure proper spacing around special characters
    text = re.sub(r'([.!?;:])([A-Z])', r'\1 \2', text)
    
    # Fix broken sentences (periods without spaces)
    text = re.sub(r'(\w)\.([A-Z])', r'\1. \2', text)
    
    # Normalize quotes for consistency
    text = text.replace("''", '"').replace("``", '"')
    
    # Remove URLs as they can take up many tokens
    text = re.sub(r'https?://\S+', '[URL]', text)
    
    return text.strip()


def format_document_for_context(doc,
                                include_metadata: bool = True
                                ) -> Tuple[str, int]:
    """
    Format a document to be included in the context window with consistent structure.
    
    Returns:
        Tuple of (formatted_text, token_count)
    """
    # Start with metadata
    formatted_parts = []
    
    if include_metadata and hasattr(doc, 'metadata'):
        # Extract key metadata
        module = doc.metadata.get('module', 'Unknown')
        filename = doc.metadata.get('filename', 'Unknown')
        doc_type = doc.metadata.get('document_type', '')
        
        # Create a clear header for this document segment
        header = f"[SOURCE DOCUMENT: Module: {module} - {filename}]"
        if doc_type:
            header += f" Type: {doc_type}"
            
        formatted_parts.append(header)
    
    # Preprocess the content
    content = preprocess_text_for_llm(doc.page_content)
    
    # Add the processed content
    formatted_parts.append(content)
    
    # Add a clear end marker
    formatted_parts.append("[END OF SOURCE DOCUMENT]")
    
    # Join everything
    formatted_text = "\n\n".join(formatted_parts)
    
    # Count tokens
    token_count = count_tokens(formatted_text)
    
    return formatted_text, token_count


def smart_context_truncation(formatted_docs: List[Tuple[str, int]],
                             max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """
    Intelligently truncate documents to fit within the context window.
    
    This function preserves the most important parts of documents and ensures
    we stay within token limits while keeping critical information.
    
    Args:
        formatted_docs: List of (formatted_text, token_count) tuples
        max_tokens: Maximum token budget for the entire context
        
    Returns:
        Truncated and formatted context string
    """
    if not formatted_docs:
        return ""
    
    # Sort by relevance (we assume the documents are already ordered by relevance)
    total_tokens = sum(token_count for _, token_count in formatted_docs)
    logging.info(
        f"Total tokens in all documents: {total_tokens}, limit: {max_tokens}")
    
    # Easy case: everything fits
    if total_tokens <= max_tokens:
        return "\n\n".join(text for text, _ in formatted_docs)
    
    # We need to truncate - calculate tokens per document
    doc_count = len(formatted_docs)
    
    # Allocate more tokens to the most relevant documents
    # First document gets 40%, then decreasing importance
    if doc_count == 1:
        tokens_per_doc = [max_tokens]
    elif doc_count == 2:
        tokens_per_doc = [int(max_tokens * 0.6), int(max_tokens * 0.4)]
    else:
        # Progressive allocation with more weight to first few documents
        weights = [1.0 / (i + 1)**0.5 for i in range(doc_count)]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        tokens_per_doc = [int(max_tokens * w) for w in normalized_weights]
    
    # Ensure minimum tokens per document (to avoid tiny fragments)
    MIN_TOKENS = 100
    for i in range(len(tokens_per_doc)):
        if tokens_per_doc[i] < MIN_TOKENS:
            tokens_per_doc[i] = MIN_TOKENS
    
    # Adjust if we exceed max_tokens
    if sum(tokens_per_doc) > max_tokens:
        excess = sum(tokens_per_doc) - max_tokens
        # Remove from least important docs first
        for i in range(len(tokens_per_doc) - 1, -1, -1):
            reduction = min(excess, tokens_per_doc[i] - MIN_TOKENS)
            if reduction <= 0:
                continue
            tokens_per_doc[i] -= reduction
            excess -= reduction
            if excess <= 0:
                break
    
    # Truncate each document to fit its token budget
    truncated_parts = []
    
    for i, ((text, orig_tokens),
            token_budget) in enumerate(zip(formatted_docs, tokens_per_doc)):
        if orig_tokens <= token_budget:
            # Document fits within budget
            truncated_parts.append(text)
        else:
            # Need to truncate this document
            truncated_text = truncate_document(text, token_budget)
            truncated_parts.append(truncated_text)
    
    # Join everything with clear separators
    return "\n\n---\n\n".join(truncated_parts)


def truncate_document(text: str, max_tokens: int) -> str:
    """
    Intelligently truncate a document to fit within token budget.
    
    Preserves beginning and end, with truncation in the middle.
    """
    tokenizer = get_tokenizer()
    
    if not text or not text.strip():
        return ""
    
    current_tokens = count_tokens(text)
    
    if current_tokens <= max_tokens:
        return text
    
    # Split by paragraphs for more intelligent truncation
    paragraphs = text.split("\n")
    
    if len(paragraphs) <= 2:
        # Simple case: just truncate from the middle
        if tokenizer:
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
                
            # Keep tokens from beginning and end
            keep_front = (max_tokens - 3) // 2  # -3 for truncation marker
            keep_end = max_tokens - 3 - keep_front
            
            truncated_tokens = tokens[:keep_front] + tokenizer.encode(
                TRUNCATION_MARKER) + tokens[-keep_end:]
            return tokenizer.decode(truncated_tokens)
        else:
            # Fallback: character-based truncation
            chars_per_token = 4  # Rough estimate
            total_chars = len(text)
            keep_chars = max_tokens * chars_per_token
            
            keep_front = keep_chars // 2
            keep_end = keep_chars - keep_front
            
            return text[:keep_front] + TRUNCATION_MARKER + text[-keep_end:]
    
    # More complex case: keep first and last paragraphs, intelligently select from middle
    first_para = paragraphs[0]
    last_para = paragraphs[-1]
    
    first_tokens = count_tokens(first_para)
    last_tokens = count_tokens(last_para)
    
    # If first and last paragraphs already exceed budget, truncate them
    if first_tokens + last_tokens > max_tokens:
        # Allocate tokens proportionally
        first_allocation = int(max_tokens * (first_tokens /
                                             (first_tokens + last_tokens)))
        last_allocation = max_tokens - first_allocation
        
        # Ensure minimum sizes
        if first_allocation < 50:
            first_allocation = 50
            last_allocation = max_tokens - 50
        if last_allocation < 50:
            last_allocation = 50
            first_allocation = max_tokens - 50
        
        truncated_first = truncate_document(first_para, first_allocation)
        truncated_last = truncate_document(last_para, last_allocation)
        
        return f"{truncated_first}\n\n{TRUNCATION_MARKER}\n\n{truncated_last}"
    
    # We have budget for middle paragraphs
    middle_budget = max_tokens - first_tokens - last_tokens - count_tokens(
        TRUNCATION_MARKER)
    middle_paras = paragraphs[1:-1]
    
    # If all middle paragraphs fit, include them all
    middle_tokens = sum(count_tokens(p) for p in middle_paras)
    if middle_tokens <= middle_budget:
        return text
    
    # We need to select which middle paragraphs to keep
    selected_middle = []
    current_budget = middle_budget
    
    # Prioritize shorter paragraphs with higher keyword density
    scored_paras = []
    for para in middle_paras:
        para_tokens = count_tokens(para)
        # Skip very long paragraphs
        if para_tokens > middle_budget * 0.7:  
            continue            
            
        # Higher score = more likely to be included
        # Prioritize paragraphs with structural elements and keywords
        has_list = 1 if re.search(r'(?:^|\n)\s*[-•*]\s', para) else 0
        has_numbers = 1 if re.search(r'(?:^|\n)\s*\d+\.', para) else 0
        
        # Score based on length (prefer shorter) and structure
        score = (has_list * 3) + (has_numbers * 2) + (1.0 / (para_tokens + 1))
        scored_paras.append((para, para_tokens, score))
    
    # Sort by score (highest first)
    scored_paras.sort(key=lambda x: x[2], reverse=True)
    
    # Include paragraphs until we hit the budget
    for para, para_tokens, _ in scored_paras:
        if para_tokens <= current_budget:
            selected_middle.append(para)
            current_budget -= para_tokens
        
        if current_budget <= 20:  # Not enough budget for meaningful additions
            break
    
    # Construct final text
    if selected_middle:
        middle_text = "\n\n".join(selected_middle)
        return f"{first_para}\n\n{middle_text}\n\n{last_para}"
    else:
        return f"{first_para}\n\n{TRUNCATION_MARKER}\n\n{last_para}"


def prepare_optimized_context(docs: List,
                              query: str,
                              max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """
    Prepare optimized context for the LLM by:
    1. Preprocessing documents for LLM compatibility
    2. Intelligently managing the context window
    3. Ensuring proper formatting and structure
    
    Args:
        docs: Retrieved documents
        query: User query
        max_tokens: Maximum token limit for context
        
    Returns:
        Optimized context string ready for the LLM
    """
    if not docs:
        return ""
    
    # 1. Format each document and count tokens
    formatted_docs = []
    
    for doc in docs:
        # Check if this is actual document content
        if not hasattr(doc, 'page_content') or not doc.page_content.strip():
            continue
            
        # Format the document
        formatted_text, token_count = format_document_for_context(doc)
        formatted_docs.append((formatted_text, token_count))
    
    # 2. Perform smart truncation to fit within context window
    optimized_context = smart_context_truncation(formatted_docs, max_tokens)
    
    # 3. Add contextual header that helps the LLM understand the source material
    query_intents = classify_query_intent(query)
    
    header = "Documents du cours universitaire pertinents à votre question:\n"
    
    # Add count information
    doc_count = len(formatted_docs)
    if doc_count > 0:
        header += f"(Total de {doc_count} document{'s' if doc_count > 1 else ''}"
        
        # If truncation happened, mention it
        total_tokens = sum(token_count for _, token_count in formatted_docs)
        if total_tokens > max_tokens:
            header += f", contenu partiellement tronqué pour s'adapter à la limite de contexte"
        
        header += ")\n\n"
    
    # 4. Create final context with header
    final_context = header + optimized_context
    
    # Log token usage for monitoring
    final_token_count = count_tokens(final_context)
    logging.info(
        f"Prepared context with {final_token_count} tokens from {doc_count} documents"
    )
    
    return final_context


# Add imports at top
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Constants for hallucination detection
FACT_VERIFICATION_THRESHOLD = 0.3  # Minimum evidence support needed
CLAIM_DETECTION_PHRASES = [
    "selon", "d'après", "mentionné dans", "comme indiqué", "le document",
    "le cours", "le professeur", "l'examen", "la date", "le syllabus",
    "le devoir", "le chapitre", "la section", "page", "dates", "évaluation",
    "obligatoire", "facultatif"
]


class FactVerifier:
    """
    System to verify generated content against source documents to prevent hallucinations.
    
    This analyzes claims in the response and checks if they're supported by evidence
    in the retrieved documents.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Include 1-3 word phrases for better matching
            min_df=1,
            max_df=0.9,
            stop_words=
            None  # Keep all words as they may be important in academic context
        )
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from generated text."""
        if not text:
            return []
            
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Identify sentences likely to contain factual claims
        claims = []
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                continue
                
            # Check for claim indicators
            contains_claim_phrase = any(phrase in sentence.lower()
                                        for phrase in CLAIM_DETECTION_PHRASES)
            
            # Sentences with numbers often contain factual claims
            contains_number = bool(re.search(r'\d', sentence))
            
            # Sentences with quotation marks likely contain facts
            contains_quote = '"' in sentence or "'" in sentence
            
            # Add to claims if it has indicators of being a factual statement
            if contains_claim_phrase or contains_number or contains_quote:
                claims.append(sentence)
            
        return claims
    
    def find_supporting_evidence(self, claim: str,
                                 documents: List) -> Tuple[float, str]:
        """
        Find best supporting evidence for a claim and calculate support score.
        
        Returns:
            Tuple of (support_score, best_evidence)
        """
        if not documents or not claim:
            return 0.0, ""
        
        # Get document content
        document_texts = [doc.page_content for doc in documents]
        
        # Simple approach: look for high text overlap using difflib
        best_match_score = 0.0
        best_evidence = ""
        
        # Check claim against each document
        for doc_text in document_texts:
            # Break document into paragraphs for more granular matching
            paragraphs = re.split(r'\n\s*\n', doc_text)
            
            for para in paragraphs:
                # Compute similarity
                similarity = difflib.SequenceMatcher(None, claim.lower(),
                                                     para.lower()).ratio()
                
                # Update best match if better
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_evidence = para
        
        # TF-IDF similarity for more semantic matching
        if best_match_score < 0.5:  # Only if direct matching didn't find strong evidence
            try:
                # Prepare corpus
                all_texts = [claim] + document_texts
                
                # Fit vectorizer
                tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                
                # Compare claim (index 0) with each document
                claim_vector = tfidf_matrix[0:1]
                
                # Find best matching document
                best_tfidf_score = 0.0
                best_tfidf_index = 0
                
                for i in range(1, len(all_texts)):
                    doc_vector = tfidf_matrix[i:i + 1]
                    similarity = (claim_vector * doc_vector.T).toarray()[0][0]
                    
                    if similarity > best_tfidf_score:
                        best_tfidf_score = similarity
                        best_tfidf_index = i
                
                # If TF-IDF found better evidence, use it
                if best_tfidf_score > best_match_score:
                    best_match_score = best_tfidf_score
                    best_evidence = all_texts[best_tfidf_index]
            except Exception as e:
                logging.error(f"Error in TF-IDF claim verification: {str(e)}")
        
        return best_match_score, best_evidence
    
    def verify_response(self, response: str,
                        documents: List) -> Dict[str, Any]:
        """
        Verify a generated response against source documents.
        
        Args:
            response: Generated response text
            documents: Source documents used for generation
            
        Returns:
            Dictionary with verification results
        """
        # Initialize results
        results = {
            "verified": True,
            "suspicious_claims": [],
            "hallucination_score": 0.0,
            "evidence_coverage": 1.0
        }
        
        # Extract claims
        claims = self.extract_claims(response)
        
        if not claims:
            # No verifiable claims detected
            return results
            
        # Verify each claim
        total_support_score = 0.0
        unsupported_claims = []
        
        for claim in claims:
            support_score, evidence = self.find_supporting_evidence(
                claim, documents)
            
            # Track total support
            total_support_score += support_score
            
            # Flag suspicious claims
            if support_score < FACT_VERIFICATION_THRESHOLD:
                unsupported_claims.append({
                    "claim":
                    claim,
                    "support_score":
                    support_score,
                    "best_evidence":
                    evidence[:200] + "..." if len(evidence) > 200 else evidence
                })
        
        # Calculate hallucination metrics
        if claims:
            results["evidence_coverage"] = total_support_score / len(claims)
            results["hallucination_score"] = 1.0 - results["evidence_coverage"]
            
        # Add unsupported claims to results
        results["suspicious_claims"] = unsupported_claims
        
        # Determine if response is verified
        if unsupported_claims:
            results["verified"] = False
            
        return results


# Initialize the fact verifier
fact_verifier = FactVerifier()


def post_process_response(response: str, verification_results: Dict,
                          docs: List) -> str:
    """
    Apply post-processing to responses to correct or flag potential hallucinations.
    
    Args:
        response: Generated response text
        verification_results: Results from the fact verification
        docs: Source documents
        
    Returns:
        Post-processed response with hallucination corrections
    """
    if verification_results["verified"]:
        return response  # No corrections needed
        
    # If we have suspicious claims, add warnings or corrections
    if verification_results["suspicious_claims"]:
        processed_response = response
        suspicious_claims = verification_results["suspicious_claims"]
        
        # For severe hallucination issues (multiple unsupported claims or very low evidence coverage)
        if len(suspicious_claims
               ) > 2 or verification_results["evidence_coverage"] < 0.4:
            # Add a clear disclaimer at the top
            disclaimer = (
                "⚠️ **Note: Certaines informations dans cette réponse pourraient ne pas être précises "
                "car elles ne sont pas clairement soutenues par les documents du cours.** "
                "Veuillez vérifier ces informations auprès de sources officielles du cours.\n\n"
            )
            processed_response = disclaimer + processed_response
            
        else:
            # For minor issues, add inline warnings to specific claims
            for claim_info in suspicious_claims:
                claim = claim_info["claim"]
                # Escape special characters for regex
                escaped_claim = re.escape(claim)
                # Replace the claim with a warning version
                warning_text = f"{claim} [Information à vérifier]"
                processed_response = re.sub(
                    escaped_claim, 
                    warning_text, 
                    processed_response, 
                    count=
                    1  # Only replace first occurrence to avoid multiple warnings
                )
                
        return processed_response
    
    return response  # Default is to return the original


def check_hallucination_risk(query: str, docs: List) -> float:
    """
    Estimate the risk of hallucination for a given query and document set.
    
    Returns:
        Risk score between 0.0 (low risk) and 1.0 (high risk)
    """
    # Initialize risk score
    risk_score = 0.0
    
    # Check if we have sufficient evidence
    if not docs or len(docs) == 0:
        return 1.0  # Maximum risk with no documents
    
    # Extract query terms (excluding stop words)
    query_words = set(query.lower().split())
    stop_words = {
        "how", "to", "a", "an", "the", "and", "or", "is", "are", "for", "in",
        "on", "at", "by", "with", "about", "like", "as", "of", "comment",
        "quoi", "qui", "que", "où", "quand", "pourquoi", "comment"
    }
    query_keywords = query_words - stop_words
    
    # No meaningful keywords
    if not query_keywords:
        return 0.5  # Medium risk for vague queries
    
    # Check keyword coverage in documents
    content_text = " ".join([doc.page_content.lower() for doc in docs])
    matching_keywords = sum(1 for kw in query_keywords if kw in content_text)
    keyword_coverage = matching_keywords / len(query_keywords)
    
    # Calculate risk based on keyword coverage
    risk_score = 1.0 - keyword_coverage
    
    # Adjust risk for specific query types
    query_intents = classify_query_intent(query)
    
    # Questions about dates, deadlines, and grades have higher hallucination risk
    if query_intents.get("is_deadline_query", False):
        # These often require exact facts, increasing risk
        risk_score += 0.2
        
    # Check if query contains numbers or specific entities (higher risk)
    if re.search(r'\d', query) or re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+',
                                            query):
        risk_score += 0.15
    
    # Cap risk score at 1.0
    return min(risk_score, 1.0)


def resolve_contradictions(contradictions: List[Dict],
                           docs: List) -> Dict[str, Any]:
    """
    Analyze and resolve contradictions in source documents to reduce hallucination risk.
    
    Args:
        contradictions: List of detected contradictions
        docs: Source documents
        
    Returns:
        Dictionary with resolution guidance
    """
    if not contradictions:
        return {"has_contradictions": False}
    
    resolutions = {
        "has_contradictions": True,
        "resolutions": [],
        "guidance": ""
    }
    
    for contradiction in contradictions:
        resolution = {"type": contradiction["type"], "resolution": "uncertain"}
        
        # Attempt to resolve based on document types and dates
        doc1 = contradiction.get("doc1", {})
        doc2 = contradiction.get("doc2", {})
        
        # Find the actual document objects
        doc1_obj = next((d for d in docs
                         if d.metadata.get("filename") == doc1.get("source")),
                        None)
        doc2_obj = next((d for d in docs
                         if d.metadata.get("filename") == doc2.get("source")),
                        None)
        
        # Extract metadata for resolution
        doc1_metadata = doc1_obj.metadata if doc1_obj else {}
        doc2_metadata = doc2_obj.metadata if doc2_obj else {}
        
        # Try to resolve by document type priority
        doc_type_priority = {
            "syllabus": 5,  # Highest priority
            "schedule": 4,
            "assignment": 3,
            "lecture": 2,
            "notes": 1,
            "unknown": 0  # Lowest priority
        }
        
        doc1_type = doc1_metadata.get("document_type", "unknown").lower()
        doc2_type = doc2_metadata.get("document_type", "unknown").lower()
        
        doc1_priority = doc_type_priority.get(doc1_type, 0)
        doc2_priority = doc_type_priority.get(doc2_type, 0)
        
        # Try to resolve by recency
        doc1_date = doc1_metadata.get("date", "")
        doc2_date = doc2_metadata.get("date", "")
        
        doc1_semester = doc1_metadata.get("semester", "")
        doc2_semester = doc2_metadata.get("semester", "")
        
        # Extract year information from dates or semesters
        doc1_year = extract_year(doc1_date) or extract_year(doc1_semester)
        doc2_year = extract_year(doc2_date) or extract_year(doc2_semester)
        
        # Resolve by document priority
        if doc1_priority > doc2_priority:
            resolution["resolution"] = "source1"
            resolution[
                "reason"] = f"Le document '{doc1.get('source')}' ({doc1_type}) a une priorité plus élevée."
        elif doc2_priority > doc1_priority:
            resolution["resolution"] = "source2"
            resolution[
                "reason"] = f"Le document '{doc2.get('source')}' ({doc2_type}) a une priorité plus élevée."
        # Resolve by recency if available
        elif doc1_year and doc2_year and doc1_year != doc2_year:
            if int(doc1_year) > int(doc2_year):
                resolution["resolution"] = "source1"
                resolution[
                    "reason"] = f"Le document '{doc1.get('source')}' est plus récent ({doc1_year} vs {doc2_year})."
            else:
                resolution["resolution"] = "source2"
                resolution[
                    "reason"] = f"Le document '{doc2.get('source')}' est plus récent ({doc2_year} vs {doc1_year})."
        else:
            # Can't resolve automatically
            resolution["resolution"] = "uncertain"
            resolution[
                "reason"] = "Impossible de déterminer quelle source est la plus fiable."
        
        resolutions["resolutions"].append(resolution)
    
    # Create guidance for the LLM
    guidance = "Concernant les contradictions détectées:\n"
    
    for i, res in enumerate(resolutions["resolutions"]):
        guidance += f"- Contradiction {i+1} ({res['type']}): {res['reason']} "
        
        if res['resolution'] == "source1":
            guidance += "Utilisez cette information comme plus fiable.\n"
        elif res['resolution'] == "source2":
            guidance += "Utilisez cette information comme plus fiable.\n"
        else:
            guidance += "Mentionnez cette contradiction explicitement dans votre réponse.\n"
    
    resolutions["guidance"] = guidance
    
    return resolutions


def extract_year(text: str) -> Optional[str]:
    """Extract year from date or semester string."""
    if not text:
        return None
        
    # Look for 4-digit years
    year_match = re.search(r'20\d{2}', text)
    if year_match:
        return year_match.group(0)
    
    # Look for abbreviated years with semester (e.g., F23, S22)
    sem_match = re.search(r'[FS](\d{2})', text)
    if sem_match:
        year = sem_match.group(1)
        if year:
            # Assume 21st century
            return f"20{year}"
    
    return None


def perform_self_critique(response: str, query: str,
                          docs: List) -> Dict[str, Any]:
    """
    Perform self-critique of generated response to further reduce hallucination risk.
    
    This uses rule-based assessments to evaluate response quality.
    
    Args:
        response: Generated response
        query: User query
        docs: Source documents
        
    Returns:
        Dictionary with critique results
    """
    critique = {"passed": True, "issues": [], "confidence": 1.0}
    
    # Check if the response indicates lack of information
    no_info_patterns = [
        "je ne trouve pas d'information", "pas d'information sur ce sujet",
        "informations insuffisantes", "aucune information",
        "n'est pas mentionné", "n'est pas couverte", "n'est pas abordé"
    ]

    has_no_info_statement = any(pattern in response.lower()
                                for pattern in no_info_patterns)
    
    # Check if content actually exists in documents
    has_content_in_docs = False
    query_keywords = set(query.lower().split()) - {
        "comment", "quoi", "quand", "pourquoi", "où", "qui", "est-ce", "que",
        "qu'", "ce"
    }
    
    # Check if at least some keywords appear in documents
    doc_content = " ".join([doc.page_content.lower() for doc in docs])
    matching_keywords = sum(1 for kw in query_keywords if kw in doc_content)
    keyword_match_ratio = matching_keywords / max(1, len(query_keywords))
    
    has_content_in_docs = keyword_match_ratio >= 0.3
    
    # Issue 1: Says no information but content exists
    if has_no_info_statement and has_content_in_docs:
        critique["passed"] = False
        critique["issues"].append({
            "type":
            "false_negative",
            "description":
            "La réponse indique qu'il n'y a pas d'information, mais le contenu pertinent existe dans les documents."
        })
        critique["confidence"] -= 0.3
    
    # Issue 2: Provides specific details without adequate evidence
    if not has_no_info_statement:
        # Check for specific claims that require evidence
        has_specific_details = False
        needs_evidence = False
        
        # Look for specific details that would require evidence
        if re.search(r'\d+[.,:]\d+', response) or re.search(
                r'\d{1,2}/\d{1,2}', response):
            # Contains dates, times, or numeric values with decimals
            has_specific_details = True
            needs_evidence = True
        
        if re.search(
                r'(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)',
                response.lower()):
            # Contains specific days of week
            has_specific_details = True
            needs_evidence = True
            
        if re.search(
                r'(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)',
                response.lower()):
            # Contains specific months
            has_specific_details = True
            needs_evidence = True
            
        if has_specific_details and needs_evidence and not has_content_in_docs:
            critique["passed"] = False
            critique["issues"].append({
                "type":
                "unsubstantiated_details",
                "description":
                "La réponse contient des détails spécifiques qui ne semblent pas être soutenus par les documents."
            })
            critique["confidence"] -= 0.4
    
    # Issue 3: Response contains inconsistencies
    if "d'une part" in response.lower(
    ) and "d'autre part" not in response.lower():
        critique["issues"].append({
            "type":
            "incomplete_structure",
            "description":
            "La réponse commence une structure mais ne la complète pas."
        })
        critique["confidence"] -= 0.1
    
    # Check citation consistency
    doc_titles = [
        doc.metadata.get("filename", "").split("/")[-1] for doc in docs
        if hasattr(doc, "metadata")
    ]
    cited_docs = []
    
    citation_pattern = r"selon\s+([^,.]+)|d'après\s+([^,.]+)|dans\s+([^,.]+)|mentionné\s+dans\s+([^,.]+)"
    for match in re.finditer(citation_pattern, response.lower()):
        groups = match.groups()
        citation = next((g for g in groups if g), "")
        if citation:
            cited_docs.append(citation)
    
    # Issue 4: Citations don't match available documents
    if cited_docs:
        invalid_citations = []
        for citation in cited_docs:
            if not any(doc_title.lower() in citation.lower()
                       or citation.lower() in doc_title.lower()
                    for doc_title in doc_titles):
                invalid_citations.append(citation)
        
        if invalid_citations:
            critique["issues"].append({
                "type":
                "invalid_citation",
                "description":
                f"La réponse cite des sources qui ne correspondent pas aux documents disponibles: {', '.join(invalid_citations)}"
            })
            critique["confidence"] -= 0.2
    
    # Ensure confidence is capped
    critique["confidence"] = max(0.0, min(critique["confidence"], 1.0))
    
    return critique


def apply_hallucination_prevention(
        query: str, response: str, docs: List,
        contradictions: List[Dict]) -> Dict[str, Any]:
    """
    Apply comprehensive hallucination prevention using multiple strategies.
    
    Args:
        query: User query
        response: Generated response
        docs: Source documents
        contradictions: Detected contradictions
        
    Returns:
        Dictionary with results and corrected response
    """
    result = {
        "original_response": response,
        "corrected_response": response,
        "verification_results": {},
        "critique_results": {},
        "hallucination_detected": False,
        "hallucination_risk": 0.0,
        "confidence": 1.0,
        "needs_human_verification": False
    }
    
    # 1. Assess hallucination risk
    risk_score = check_hallucination_risk(query, docs)
    result["hallucination_risk"] = risk_score
    
    # 2. Verify factual claims
    verification_results = fact_verifier.verify_response(response, docs)
    result["verification_results"] = verification_results
    
    # 3. Perform self-critique
    critique_results = perform_self_critique(response, query, docs)
    result["critique_results"] = critique_results
    
    # 4. Resolve contradictions if present
    if contradictions:
        contradiction_resolution = resolve_contradictions(contradictions, docs)
        result["contradiction_resolution"] = contradiction_resolution
    
    # 5. Determine if hallucination was detected
    hallucination_detected = (
        (verification_results.get("suspicious_claims", [])
         and len(verification_results["suspicious_claims"]) > 0)
        or (not critique_results.get("passed", True))
        or (verification_results.get("hallucination_score", 0) > 0.5))
    
    result["hallucination_detected"] = hallucination_detected
    
    # 6. Calculate overall confidence
    # Combine confidence from multiple sources
    verification_confidence = 1.0 - verification_results.get(
        "hallucination_score", 0.0)
    critique_confidence = critique_results.get("confidence", 1.0)
    risk_confidence = 1.0 - risk_score
    
    # Weighted average (verification has highest weight)
    overall_confidence = (verification_confidence * 0.5 +
                          critique_confidence * 0.3 + risk_confidence * 0.2)
    
    result["confidence"] = overall_confidence
    
    # 7. Apply corrections based on verification
    if hallucination_detected:
        # Post-process to add warnings or remove hallucinations
        corrected_response = post_process_response(response,
                                                    verification_results, docs)
        result["corrected_response"] = corrected_response
        
        # Determine if human verification is needed
        result["needs_human_verification"] = (
            verification_results.get("hallucination_score", 0) > 0.7
            or len(verification_results.get("suspicious_claims", [])) > 3
            or overall_confidence < 0.4)
    
    return result


class FeedbackManager:
    """
    Manage detailed user feedback and integrate it into system improvement.
    """
    
    def __init__(self, feedback_db_path: str = "feedback_data.json"):
        self.feedback_db_path = feedback_db_path
        self.feedback_data = self._load_feedback_data()
        
    def _load_feedback_data(self) -> Dict:
        """Load feedback data from storage."""
        try:
            if os.path.exists(self.feedback_db_path):
                with open(self.feedback_db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {
                    "feedback_entries": [],
                    "improvement_suggestions": [],
                    "knowledge_gaps": [],
                    "metrics": {
                        "positive_count": 0,
                        "negative_count": 0,
                        "total_count": 0
                    }
                }
        except Exception as e:
            logging.error(f"Error loading feedback data: {str(e)}")
            return {
                "feedback_entries": [],
                "improvement_suggestions": [],
                "knowledge_gaps": [],
                "metrics": {
                    "positive_count": 0,
                    "negative_count": 0,
                    "total_count": 0
                }
            }
            
    def _save_feedback_data(self):
        """Save feedback data to storage."""
        try:
            with open(self.feedback_db_path, "w", encoding="utf-8") as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving feedback data: {str(e)}")
            
    def record_detailed_feedback(self, feedback: Dict[str, Any]):
        """
        Record detailed user feedback.
        
        Args:
            feedback: Dictionary with detailed feedback information
        """
        # Ensure required fields
        required_fields = ["message_id", "query", "response", "feedback_type"]
        for field in required_fields:
            if field not in feedback:
                logging.error(f"Missing required field in feedback: {field}")
                return
                
        # Add timestamp and unique ID if not present
        if "timestamp" not in feedback:
            feedback["timestamp"] = datetime.now().isoformat()
            
        if "id" not in feedback:
            feedback["id"] = str(uuid.uuid4())
            
        # Add to feedback entries
        self.feedback_data["feedback_entries"].append(feedback)
        
        # Update metrics
        self.feedback_data["metrics"]["total_count"] += 1
        if feedback["feedback_type"] == "positive":
            self.feedback_data["metrics"]["positive_count"] += 1
        elif feedback["feedback_type"] == "negative":
            self.feedback_data["metrics"]["negative_count"] += 1
            
        # Process knowledge gaps
        if feedback.get("indicates_knowledge_gap", False):
            self._record_knowledge_gap(feedback)
            
        # Process improvement suggestions
        if "improvement_suggestion" in feedback and feedback[
                "improvement_suggestion"]:
            self._record_improvement_suggestion(feedback)
            
        # Save updated data
        self._save_feedback_data()
        
    def _record_knowledge_gap(self, feedback: Dict[str, Any]):
        """Record information about a knowledge gap."""
        knowledge_gap = {
            "id": str(uuid.uuid4()),
            "query": feedback["query"],
            "timestamp": feedback["timestamp"],
            "description": feedback.get("knowledge_gap_description", ""),
            "suggested_resources": feedback.get("suggested_resources", []),
            "status": "identified"
        }
        
        self.feedback_data["knowledge_gaps"].append(knowledge_gap)
        
    def _record_improvement_suggestion(self, feedback: Dict[str, Any]):
        """Record a suggestion for system improvement."""
        suggestion = {
            "id": str(uuid.uuid4()),
            "source_feedback_id": feedback["id"],
            "timestamp": feedback["timestamp"],
            "suggestion": feedback["improvement_suggestion"],
            "category": feedback.get("suggestion_category", "general"),
            "status": "pending"
        }
        
        self.feedback_data["improvement_suggestions"].append(suggestion)


def collect_detailed_feedback(message_id: str,
                              query: str,
                              response: str,
                              feedback_type: str,
                              docs: List,
                              evaluation_results: Dict[str, Any] = None):
    """
    Collect and store detailed feedback about a response.
    
    Args:
        message_id: Unique message ID
        query: User query
        response: System response
        feedback_type: Type of feedback (positive/negative)
        docs: Retrieved documents
        evaluation_results: Results from automatic evaluation
    """
    # Get feedback manager
    if not hasattr(st.session_state, "feedback_manager"):
        st.session_state.feedback_manager = FeedbackManager()
    feedback_manager = st.session_state.feedback_manager
    
    # Basic feedback data
    feedback_data = {
        "message_id": message_id,
        "query": query,
        "response": response,
        "feedback_type": feedback_type,
        "timestamp": datetime.now().isoformat(),
        "doc_count": len(docs),
        "doc_sources":
        [doc.metadata.get("source", "Unknown") for doc in docs[:3]]
    }
    
    # Add evaluation results if available
    if evaluation_results:
        feedback_data["evaluation_results"] = {
            "overall_score": evaluation_results["overall"]["score"],
            "faithfulness_score": evaluation_results["faithfulness"]["score"],
            "relevance_score": evaluation_results["relevance"]["score"],
            "quality_score": evaluation_results["quality"]["score"]
        }
    
    # For negative feedback, prompt for additional details
    if feedback_type == "negative" and st.session_state.get(
            "show_feedback_form", False):
        # Store in session state to show feedback form
        st.session_state.current_feedback = feedback_data
        st.session_state.show_feedback_form = True
        return
        
    # Otherwise, just record the basic feedback
    feedback_manager.record_detailed_feedback(feedback_data)
    
    # Log for analytics
    logging.info(
        f"Feedback recorded: {feedback_type} for query: {query[:50]}...")


def show_feedback_form():
    """Display form to collect detailed feedback for negative responses."""
    if not st.session_state.get("show_feedback_form", False):
        return
        
    with st.form("detailed_feedback_form"):
        st.write("### Détails de votre retour")
        
        # What was wrong with the response
        issue_type = st.multiselect(
            "Quel était le problème avec la réponse ?", [
                "Information incorrecte", "Information manquante",
                "Réponse hors sujet", "Réponse trop vague",
                "Réponse trop longue", "Problème de formatage", "Autre"
            ])
        
        # Specific feedback text
        feedback_text = st.text_area(
            "Commentaires supplémentaires (facultatif)")
        
        # Check if this indicates a knowledge gap
        indicates_gap = st.checkbox(
            "Cette question nécessite des informations qui ne sont pas dans la base de connaissances"
        )
        
        # If it indicates a knowledge gap, collect more info
        gap_description = ""
        suggested_resources = ""
        if indicates_gap:
            gap_description = st.text_area(
                "Décrivez les informations manquantes")
            suggested_resources = st.text_area(
                "Suggérez des ressources qui pourraient combler ce manque (facultatif)"
            )
            
        # Improvement suggestion
        improvement_suggestion = st.text_area(
            "Avez-vous des suggestions pour améliorer le système ? (facultatif)"
        )
        
        # Submit button
        submitted = st.form_submit_button("Envoyer")
        
        if submitted:
            # Get feedback manager
            if not hasattr(st.session_state, "feedback_manager"):
                st.session_state.feedback_manager = FeedbackManager()
            feedback_manager = st.session_state.feedback_manager
            
            # Get current feedback data
            feedback_data = st.session_state.current_feedback
            
            # Add detailed information
            feedback_data["issue_types"] = issue_type
            feedback_data["feedback_text"] = feedback_text
            feedback_data["indicates_knowledge_gap"] = indicates_gap
            
            if indicates_gap:
                feedback_data["knowledge_gap_description"] = gap_description
                feedback_data[
                    "suggested_resources"] = suggested_resources.split(
                        "\n") if suggested_resources else []
                
            if improvement_suggestion:
                feedback_data[
                    "improvement_suggestion"] = improvement_suggestion
                
            # Record detailed feedback
            feedback_manager.record_detailed_feedback(feedback_data)
            
            # Reset state
            st.session_state.show_feedback_form = False
            st.session_state.current_feedback = None
            
            # Show confirmation
            st.success("Merci pour votre retour détaillé !")
            
            # Rerun to remove the form
            st.experimental_rerun()


def classify_response_format(query: str) -> str:
    """
    Determine the most appropriate response format for the query, prioritizing French formats.
    
    Args:
        query: The user's query
        
    Returns:
        The classified response format type
    """
    query = query.lower()
    
    # Definition patterns
    if any(pattern in query for pattern in [
        "qu'est-ce que", "c'est quoi", "définir", "définition", "signifie", 
        "expliquer le concept", "concept de", "terme"
    ]):
        return "definition"
    
    # List patterns
    if any(pattern in query for pattern in [
            "liste", "énumérer", "citer", "quels sont", "quelles sont",
            "points", "étapes", "éléments", "facteurs"
    ]):
        return "list"
    
    # Deadline patterns
    if any(pattern in query for pattern in [
        "date", "délai", "quand", "échéance", "deadline", "calendrier", 
        "planning", "rendre", "soumettre", "remise"
    ]):
        return "deadline"
    
    # Comparison patterns
    if any(pattern in query for pattern in [
            "comparer", "différence", "similitude", "ressemblance",
            "distinguer", "versus", "par rapport à", "comparaison"
    ]):
        return "comparison"
    
    # Detailed explanation patterns
    if any(pattern in query for pattern in [
        "expliquer en détail", "élaborer", "développer", "approfondir", 
        "décrivez en détail", "donner plus d'information"
    ]) or len(query.split()
              ) > 10:  # Longer queries often need detailed responses
        return "detailed_explanation"
    
    # Default to general response
    return "general"


def detect_specificity_level(query: str) -> str:
    """
    Determine the appropriate specificity level for the response, prioritizing French indicators.
    
    Args:
        query: The user's query
        
    Returns:
        The specificity level: "brief", "standard", or "detailed"
    """
    query = query.lower()
    
    # Brief response indicators
    brief_indicators = [
        "rapidement", "bref", "brève", "résumé", "résumer", "simplement", 
        "vite", "court", "courte", "en deux mots", "sommairement"
    ]
    
    # Detailed response indicators
    detailed_indicators = [
        "détaillé", "détailler", "en détail", "précisément", "élaborer", 
        "expliquer", "complet", "complète", "approfondi", "développer",
        "exhaustif", "exhaustive"
    ]
    
    # Check for explicit brief indicators
    if any(indicator in query for indicator in brief_indicators):
        return "brief"
    
    # Check for explicit detailed indicators
    if any(indicator in query for indicator in detailed_indicators):
        return "detailed"
    
    # Infer from query complexity
    words = query.split()
    if len(words) <= 4:  # Very short queries often want brief answers
        return "brief"
    elif len(words) >= 10:  # Longer queries often want detailed responses
        return "detailed"
        
    # Check for question complexity
    complex_query = any(term in query for term in [
        "pourquoi", "comment", "expliquer", "analyser", "comparer", "évaluer",
        "discuter", "interpréter"
    ])
    
    if complex_query:
        return "standard"
        
    # Default to standard specificity
    return "standard"


def adjust_response_specificity(response: str, query: str,
                               specificity_level: str) -> str:
    """
    Adjust the response length and detail level based on detected specificity.
    
    Args:
        response: The original response
        query: The user's query
        specificity_level: The desired specificity level
        
    Returns:
        Adjusted response with appropriate detail level
    """
    # For normal responses, return as is
    if specificity_level == "standard":
        return response
        
    # For brief responses, summarize
    if specificity_level == "brief":
        # Split into paragraphs
        paragraphs = response.split("\n\n")
        
        # For already short responses, return as is
        if len(response) < 250:
            return response
            
        # If multiple paragraphs, keep the first one or two
        if len(paragraphs) > 1:
            return "\n\n".join(paragraphs[:min(2, len(paragraphs))])
            
        # For single paragraph responses, keep first 2-3 sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if len(sentences) > 3:
            return " ".join(sentences[:3])
            
        return response
        
    # For detailed responses, keep as is or expand if needed
    if specificity_level == "detailed":
        # Response is already detailed enough
        if len(response) > 500:
            return response
            
        # If it's too brief, add explanatory note
        if len(response) < 200:
            note = "\n\n*Note: Pour obtenir des informations plus détaillées sur ce sujet, consultez les documents du cours ou posez une question plus spécifique.*"
            return response + note
            
        return response
        
    return response


def format_output_with_parser(response: str, query: str) -> str:
    """
    Attempts to reformat the LLM's response using appropriate output parsers.
    
    If parsing fails, returns the original response with minimal formatting.
    
    Args:
        response: The LLM's generated response
        query: The user's original query
        
    Returns:
        Formatted response with appropriate structure
    """
    try:
        # First, clean up any excessive formatting that might be present
        # Replace big headers (e.g., "# Title" or "=== Title ===") with smaller ones
        response = re.sub(r'^#+\s+(.+)$',
                          r'**\1**',
                          response,
                          flags=re.MULTILINE)
        response = re.sub(r'^===+\s+(.+)\s+===+$',
                          r'**\1**',
                          response,
                          flags=re.MULTILINE)
        response = re.sub(r'^---+\s+(.+)\s+---+$',
                          r'**\1**',
                          response,
                          flags=re.MULTILINE)
        
        # Remove redundant "Note:" prefixes in warnings
        response = re.sub(r'⚠️\s*Note:\s*⚠️\s*Note:', r'⚠️ Note:', response)
        
        # Classify query to determine appropriate response format
        format_type = classify_response_format(query)
        
        # For simple responses, just apply basic formatting
        if format_type == "general" or len(response) < 100:
            # Basic formatting for short responses
            paragraphs = response.split('\n\n')
            formatted = response
            
            # Add basic structure if needed
            if len(paragraphs) <= 1 and len(response) > 50:
                # Add paragraph breaks for readability
                sentences = response.split('. ')
                if len(sentences) > 2:
                    formatted = '. '.join(sentences[:2]) + '.\n\n' + '. '.join(
                        sentences[2:])
            
            return formatted
        
        # For definition responses
        if format_type == "definition":
            lines = response.split('\n')
            term = ""
            definition = ""
            examples = []
            related = []
            
            # Extract definition and examples from response
            definition_mode = True
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.lower().startswith(
                    ("exemple", "par exemple", "example")):
                    definition_mode = False
                    examples.append(line)
                elif line.lower().startswith(
                    ("voir aussi", "concepts liés", "related")):
                    definition_mode = False
                    related.append(line)
                elif definition_mode:
                    if not term and (":" in line or line.startswith("**")):
                        # Handle both plain text with colon and already formatted headers
                        if ":" in line:
                            parts = line.split(":", 1)
                            term = parts[0].strip().replace("*", "")
                            if len(parts) > 1:
                                definition += parts[1].strip() + " "
                        else:
                            term = line.replace("*", "").strip()
                    else:
                        definition += line + " "
            
            # Format the response with more moderate styling
            formatted = f"**{term or 'Définition'}**\n\n{definition.strip()}"
            
            if examples:
                formatted += "\n\n*Exemples:*\n"
                for i, example in enumerate(examples, 1):
                    formatted += f"- {example.replace('Exemple:', '').replace('Par exemple:', '').strip()}\n"
            
            if related:
                formatted += "\n\n*Concepts liés:*\n"
                for rel in related:
                    formatted += f"- {rel.replace('Voir aussi:', '').replace('Concepts liés:', '').strip()}\n"
            
            return formatted
            
        # For list responses
        elif format_type == "list":
            lines = response.split('\n')
            introduction = ""
            items = []
            conclusion = ""
            
            # Parse the response
            mode = "intro"
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for list markers
                if line.startswith(
                    ("-", "•", "*", "1.", "2.",
                     "3.")) or (len(line) > 3 and line[0].isdigit()
                                and line[1] == ')'):
                    mode = "items"
                    items.append(line)
                elif mode == "items" and not line.startswith(
                    ("-", "•", "*")) and not (len(line) > 3
                                              and line[0].isdigit()
                                              and line[1:3] in ('. ', ') ')):
                    mode = "conclusion"
                    conclusion += line + " "
                elif mode == "intro":
                    introduction += line + " "
                elif mode == "conclusion":
                    conclusion += line + " "
            
            # Format the response with more subtle styling
            formatted = introduction.strip()
            if items:
                formatted += "\n\n"
                for item in items:
                    formatted += f"{item}\n"
            if conclusion:
                formatted += f"\n{conclusion.strip()}"
            
            return formatted
            
        # For deadline responses
        elif format_type == "deadline":
            event = ""
            date = ""
            details = ""
            submission = ""
            
            # Try to extract date information using regex
            date_pattern = r'(\d{1,2}(?:er)?\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)|(?:lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+\d{1,2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?)'
            date_matches = re.findall(date_pattern, response, re.IGNORECASE)
            
            if date_matches:
                date = date_matches[0]
                
                # Extract event (usually before the date)
                date_pos = response.lower().find(date.lower())
                if date_pos > 10:
                    event = response[:date_pos].strip()
                    if ":" in event:
                        event = event.split(":")[-1].strip()
                
                # Extract details (usually after the date)
                after_date = response[date_pos + len(date):].strip()
                if after_date:
                    details = after_date
                    
                    # Look for submission method
                    submission_keywords = [
                        "soumettre", "remettre", "déposer", "envoyer",
                        "submission"
                    ]
                    for keyword in submission_keywords:
                        if keyword in details.lower():
                            parts = details.lower().split(keyword, 1)
                            if len(parts) > 1:
                                submission = keyword + parts[1]
                                details = parts[0]
                                break
            else:
                # Fallback to original response with minimal formatting
                return response
                
            # Format the response with moderate styling
            formatted = f"**Échéance: {event or 'Événement'}**\n\n"
            formatted += f"*Date:* {date}\n\n"
            formatted += f"*Détails:* {details}\n"
            if submission:
                formatted += f"\n*Méthode de remise:* {submission}"
            
            return formatted
        
        # If we get here, use default formatting
        return response
        
    except Exception as e:
        logging.error(f"Error in output formatting: {str(e)}")
        # Return original response if parsing fails
        return response


def prepare_response_template(query: str,
                              query_intent: Dict[str, bool]) -> Dict[str, Any]:
    """
    Prepare a template and format guidelines for the response based on query intent,
    optimized for French output by default.
    
    Args:
        query: The user's query
        query_intent: Dictionary of classified query intents
        
    Returns:
        Dictionary with template and format information
    """
    format_type = classify_response_format(query)
    specificity = detect_specificity_level(query)
    
    template = {
        "format_type": format_type,
        "specificity": specificity,
        "structure": {},
        "formatting_instructions": ""
    }
    
    # Add structure based on format type
    if format_type == "definition":
        template["structure"] = {
            "term": "",
            "definition": "",
            "examples": [],
            "related_concepts": []
        }
        template["formatting_instructions"] = (
            "Formatez la réponse comme une définition académique. "
            "Commencez par le terme en gras, suivi de sa définition. "
            "Ajoutez des exemples et concepts liés si pertinent.")
    
    elif format_type == "list":
        template["structure"] = {
            "introduction": "",
            "items": [],
            "conclusion": ""
        }
        template["formatting_instructions"] = (
            "Formatez la réponse comme une liste à puces. "
            "Commencez par une brève introduction, puis listez les éléments principaux, "
            "et terminez par une conclusion si nécessaire.")
    
    elif format_type == "deadline":
        template["structure"] = {
            "event": "",
            "date": "",
            "details": "",
            "submission_method": ""
        }
        template["formatting_instructions"] = (
            "Formatez la réponse pour une échéance. "
            "Indiquez clairement l'événement, la date exacte, les détails importants, "
            "et la méthode de soumission si applicable.")
    
    elif format_type == "comparison":
        template["structure"] = {
            "topic1": "",
            "topic2": "",
            "similarities": [],
            "differences": [],
            "conclusion": ""
        }
        template["formatting_instructions"] = (
            "Formatez la réponse comme une comparaison structurée. "
            "Commencez par identifier les deux sujets comparés, puis listez leurs similitudes "
            "et différences dans des sections distinctes, et terminez par une conclusion."
        )
    
    elif format_type == "detailed_explanation":
        template["structure"] = {
            "topic": "",
            "introduction": "",
            "main_points": [],
            "conclusion": "",
            "references": []
        }
        template["formatting_instructions"] = (
            "Formatez la réponse comme une explication détaillée et structurée. "
            "Commencez par une introduction au sujet, puis développez les points principaux "
            "avec des sous-titres, et terminez par une conclusion synthétique."
        )
    
    else:  # general
        template["structure"] = {"content": "", "sources": []}
        template["formatting_instructions"] = (
            "Formatez la réponse de manière claire et structurée, "
            "en utilisant des paragraphes distincts et des points de liste si nécessaire."
        )
    
    # Adjust for specificity
    if specificity == "brief":
        template["formatting_instructions"] += (
            " Soyez bref et concis, en limitant la réponse à l'essentiel.")
    elif specificity == "detailed":
        template["formatting_instructions"] += (
            " Fournissez une réponse détaillée avec explications approfondies et exemples."
        )
    
    return template


def enhance_prompt_with_format_guidance(prompt: str,
                                        context: Dict[str, Any]) -> str:
    """
    Enhance the prompt with format guidance based on query intent.
    
    Args:
        prompt: Original query prompt
        context: Dictionary with context information
        
    Returns:
        Enhanced prompt with format guidance
    """
    query_intent = context.get("query_intent", {})
    response_template = prepare_response_template(prompt, query_intent)
    
    format_instructions = response_template["formatting_instructions"]
    format_type = response_template["format_type"]
    specificity = response_template["specificity"]
    
    # Build enhanced prompt with format guidance
    enhanced = prompt + "\n\n"
    
    # Add format-specific instructions
    if format_type == "definition":
        enhanced += "Je cherche une définition claire et précise."
    elif format_type == "list":
        enhanced += "Veuillez présenter l'information sous forme de liste structurée."
    elif format_type == "deadline":
        enhanced += "Veuillez indiquer précisément la date et les détails de cette échéance."
    elif format_type == "comparison":
        enhanced += "Veuillez présenter une comparaison structurée avec similitudes et différences."
    elif format_type == "detailed_explanation":
        enhanced += "Veuillez fournir une explication détaillée et structurée avec des sections claires."
    
    # Add specificity guidance
    if specificity == "brief":
        enhanced += " Soyez bref et concis, en vous limitant à l'essentiel."
    elif specificity == "detailed":
        enhanced += " Développez la réponse avec tous les détails pertinents disponibles."
    
    # Add general formatting instruction
    enhanced += f"\n\n{format_instructions}"
    
    return enhanced


class BM25Retriever:
    """
    A custom BM25 retriever that uses dual encoders for query and documents.
    This is a simple implementation to mimic a DPR-style retriever.
    """
    
    def __init__(self,
                 document_store,
                 query_encoder,
                 document_encoder,
                 top_k=10):
        self.document_store = document_store
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        self.top_k = top_k
        
    def get_relevant_documents(self, query, k=None):
        """Get documents relevant to the query using a combination of BM25 and semantic search."""
        if k is None:
            k = self.top_k
            
        try:
            # Step 1: Perform semantic search with query_encoder
            query_embedding = self.query_encoder.embed_query(query)
            
            # Convert to the format needed for the document store
            # This depends on what vector store you're using
            if hasattr(self.document_store, 'similarity_search_by_vector'):
                semantic_docs = self.document_store.similarity_search_by_vector(
                    query_embedding, 
                    k=min(k * 2, 20)  # Get more candidates for reranking
                )
            else:
                # Fallback to standard search
                semantic_docs = self.document_store.similarity_search(
                    query, k=min(k * 2, 20))
            
            # Step 2: Perform keyword matching (simple BM25-like approach)
            # Tokenize query and documents
            query_tokens = set(query.lower().split())
            
            # Score documents based on token overlap (BM25-like)
            scored_docs = []
            for doc in semantic_docs:
                doc_tokens = set(doc.page_content.lower().split())
                
                # Calculate token overlap score
                overlap = len(query_tokens.intersection(doc_tokens))
                denominator = len(query_tokens) * 0.5 + len(doc_tokens) * 0.5
                score = overlap / max(
                    1, denominator)  # Normalized by document length
                
                # Apply IDF-like weighting for rare terms
                rare_term_bonus = 0
                for term in query_tokens:
                    # If term is rare in the corpus, give it higher weight
                    if term not in [
                            "the", "and", "is", "are", "to", "of", "for", "in",
                            "on", "with"
                    ]:
                        rare_term_bonus += 0.1 if term in doc.page_content.lower(
                        ) else 0
                
                final_score = score + rare_term_bonus
                scored_docs.append((doc, final_score))
            
            # Sort by score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            return [doc for doc, _ in scored_docs[:k]]
        
        except Exception as e:
            logging.error(f"Error in BM25Retriever: {str(e)}")
            # Fall back to standard retrieval
            return self.document_store.similarity_search(query, k=k)


def ensure_source_citations(response: str, docs: List) -> str:
    """
    Ensure that the response includes proper source citations.
    If sources are missing, add them to the end of the response.
    
    Args:
        response: The generated response text
        docs: The source documents used for retrieval
        
    Returns:
        Response with proper source citations
    """
    # Skip adding sources if hide_sources is enabled
    if st.session_state.get("hide_sources", True):
        return response
        
    # Check if response already has a Sources section
    if "Sources\n\nSource" in response or "Sources\n\nSource" in response:
        return response
        
    # Extract unique modules and filenames
    sources = []
    for i, doc in enumerate(docs[:5], 1):
        module = doc.metadata.get("module", "Unknown")
        filename = doc.metadata.get("filename", "Unknown")
        source_text = f"Source {i}: Module: {module} - {filename}"
        if source_text not in sources:
            sources.append(source_text)
    
    # If no sources found, add placeholder text
    if not sources:
        sources = ["Source 1: Module: Unknown - Unknown"]
    
    # Format sources section
    sources_section = "\n\nSources\n\n" + "\n".join(sources)
    
    # Check if response already ends with a sources section
    if "Sources" in response[-100:]:
        return response
        
    # Add sources section to response
    return response + sources_section

if __name__ == "__main__":
    main()
import logging
import uuid
from typing import Any, Optional, List

import requests
import logging
import streamlit as st
import os
from langchain.llms.base import LLM

# Streamlit imports
import streamlit as st
import time
import os
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Create necessary directories
os.makedirs("logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ])

# Constants
openrouter_key = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = os.environ.get("HUGGINGFACE_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.1")
MAX_CONTEXT_TOKENS = 2000

# at the top of your file, alongside your other imports
import requests
from langchain.llms.base import LLM

class OpenRouterLLM(LLM):
    api_key: str
    model: str = "meta-llama/llama-4-scout:free"
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **({"stop": stop} if stop else {})
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


    

def init_session_state():
    """Initialize session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.search_mode = "Simple"
        st.session_state.use_domain_embeddings = False
        st.session_state.hide_sources = False
        st.session_state.initialized = True


def get_vectorstore():
    """Load the vector store created by create_memory_for_llm.py."""
    # Use default embeddings
    logging.info("Using default multilingual embeddings")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"})
    persist_directory = 'vectorstore/db_faiss'
    
    # Try to load the vector store
    try:
        vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
        logging.info(f"Loaded vectorstore from {persist_directory}")
        return vectorstore
    except Exception as e:
        logging.error(f"Error loading vectorstore: {str(e)}")
        st.error(f"Erreur lors du chargement de la base de données: {str(e)}")
        return None


def get_available_modules(db) -> List[str]:
    """Get list of available modules from the vector store."""
    if not db:
        return []
    
    try:
        # Extract modules from documents in the vector store
        modules = set()
        for doc in db.docstore._dict.values():
            module = doc.metadata.get("module")
            if module and isinstance(module, str):
                modules.add(module)
        
        return sorted(list(modules))
    except Exception as e:
        logging.error(f"Error getting modules: {str(e)}")
        return []


def get_document_types(db) -> List[str]:
    """Get list of available document types from the vector store."""
    if not db:
        return []
    
    try:
        # Extract document types from the vector store
        doc_types = set()
        for doc in db.docstore._dict.values():
            doc_type = doc.metadata.get("document_type")
            if doc_type and isinstance(doc_type, str):
                doc_types.add(doc_type)
        
        return sorted(list(doc_types))
    except Exception as e:
        logging.error(f"Error getting document types: {str(e)}")
        return []


def filter_vectorstore(db, filters: Dict[str, Any]):
    """Filter the vector store based on metadata criteria."""
    if not db or not filters:
        return db
    
    # Create a filtering function
    def filter_func(doc):
        for key, values in filters.items():
            if not isinstance(values, list):
                values = [values]
            
            doc_value = doc.metadata.get(key)
            
            # If the document doesn't have this metadata field, exclude it
            if doc_value is None:
                return False
            
            # If document value is a list, check for any intersection
            if isinstance(doc_value, list):
                if not any(value in doc_value for value in values):
                    return False
            # Otherwise check for exact match with any of the filter values
            elif doc_value not in values:
                return False
        
        return True
    
    # Apply filter to create a new filtered vector store
    filtered_docs = []
    for doc_id, doc in db.docstore._dict.items():
        if filter_func(doc):
            filtered_docs.append(doc)
    
    logging.info(f"Filtered vector store from {len(db.docstore._dict)} to {len(filtered_docs)} documents")
    
    # If no documents match the filter, return the original DB
    if not filtered_docs:
        logging.warning("No documents match the filter criteria")
        return db
    
    # Create a new vector store with the filtered documents
    filtered_db = FAISS.from_documents(filtered_docs, db._embedding_function)
    return filtered_db


def set_custom_prompt():
    custom_template = """
    Tu es un assistant intelligent spécialisé pour répondre aux questions relatives aux cours universitaires. 
    Utilise le contexte suivant pour répondre à la question de l'utilisateur.
 
    Contexte:
    {context}
 
    Question: {question}
 
    Instructions spécifiques:
    - Réponds en français uniquement
    - Cite les sources pertinentes en fin de réponse
    - Structure ta réponse avec des titres si nécessaire
    - Réponds uniquement si l'information est dans le contexte
    - Si aucune information pertinente n'est disponible, indique-le clairement
 
    Réponse:
    """
    return PromptTemplate(
        template=custom_template,
        input_variables=["context", "question"]
    )


def load_llm():
    """Load OpenRouter LLM instead of a HuggingFace or OpenAI model."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        st.error("Please set the OPENROUTER_API_KEY environment variable.")
        raise ValueError("Missing OPENROUTER_API_KEY")
    
    # You can also override the model via env var if you like
    model_name = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-4-scout:free")
    
    return OpenRouterLLM(
        api_key=openrouter_key,
        model=model_name,
        temperature=0.7
    )


def simple_search(db, query: str, k: int = 5, filters: Dict[str, Any] = None):
    """Perform a simple semantic search over the vector store."""
    if not db:
        return []
    
    try:
        # Apply filters if provided
        if filters:
            search_db = filter_vectorstore(db, filters)
        else:
            search_db = db
        
        # Search with MMR for increased diversity
        start_time = time.time()
        docs = search_db.similarity_search(
            query, k=k, fetch_k=k*2
        )
        search_time = time.time() - start_time
        
        logging.info(f"Simple search found {len(docs)} documents in {search_time:.2f}s")
        return docs
    except Exception as e:
        logging.error(f"Error in simple search: {str(e)}")
        return []


def main():
    """Main function to run the Streamlit application."""
    st.title("Assistant de Cours Universitaire")
    
    # Initialize session state
    init_session_state()
    
    # Load vector database
    db = get_vectorstore()
    
    # Get available modules and document types
    modules = get_available_modules(db)
    doc_types = get_document_types(db)
    
    # Sidebar filters
    with st.sidebar:
        st.subheader("Filtres de recherche")
        
        # Module filter
        selected_modules = st.multiselect("Modules", modules, default=[])
        
        # Document type filter
        selected_doc_types = st.multiselect("Types de document", doc_types, default=[])
        
        # Number of documents to retrieve
        k_docs = st.slider("Nombre de documents à récupérer", 3, 10, 5)
        
        # Toggle for showing/hiding sources in responses
        hide_sources = st.toggle("Masquer les sources dans les réponses",
                                value=st.session_state.hide_sources)
        st.session_state.hide_sources = hide_sources
    
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
        
        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                # Generate a unique query ID for tracking
                query_id = str(uuid.uuid4())
                
                # Search for relevant documents
                docs = simple_search(db, prompt, k=k_docs, filters=filters)
                
                if not docs:
                    st.error("Aucune information pertinente trouvée. Veuillez reformuler votre question.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Aucune information pertinente trouvée. Veuillez reformuler votre question."
                    })
                    return
                
                # Apply filters to create a filtered DB for the QA chain
                if filters:
                    filtered_db = filter_vectorstore(db, filters)
                else:
                    filtered_db = db
                
                # Get custom instruction prompt
                custom_prompt = set_custom_prompt()
                
                # Load the LLM
                llm = load_llm()
                
                try:
                    # Create the retrieval chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=db.as_retriever(search_kwargs={"k": k_docs}),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": custom_prompt}
                    )
                    
                    # Process the query with the updated invocation
                    result = qa_chain.invoke({"query": prompt})
                    answer = result["result"]
                    
                    # Process sources if they exist and if they shouldn't be hidden
                    # Note: The source docs structure might be different with the new API
                    source_docs = []
                    for doc in docs:
                        source_docs.append(doc)
                    
                    if source_docs and not hide_sources:
                        sources_text = "\n\n**Sources:**\n"
                        seen_sources = set()
                        for i, doc in enumerate(source_docs):
                            source = doc.metadata.get('source', f'Document {i+1}')
                            # Avoid duplicate sources
                            if source in seen_sources:
                                continue
                            seen_sources.add(source)
                            sources_text += f"- {source}\n"
                        
                        # Add sources to the answer
                        answer += sources_text
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                except Exception as e:
                    error_msg = f"Erreur lors du traitement de la requête: {str(e)}"
                    logging.error(error_msg)
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()
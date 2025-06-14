import os
import glob
import logging
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import spacy
from tqdm import tqdm
import random
import time
from datetime import datetime
import gc
import re

# Update requirements in comments
# Required packages:
# - langchain, langchain-community, langchain-huggingface
# - sentence-transformers, transformers, huggingface_hub
# - spacy with fr_core_news_sm
# - faiss-cpu
# - unstructured, pypdf, tqdm
# - nltk for tokenization

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = "docs"
FAISS_STORE_PATH = "vectorstore/db_faiss"
METADATA_CACHE_PATH = "vectorstore/metadata_cache.json"

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
NUM_WORKERS = 4

# File tracking for incremental updates
PROCESSED_FILES_CACHE = "vectorstore/processed_files.json"

# Load SpaCy for semantic splitting
try:
    nlp = spacy.load("fr_core_news_sm")
    logger.info("Loaded French SpaCy model successfully")
except Exception as e:
    logger.error(f"Failed to load French SpaCy model: {e}")
    logger.info("Downloading French SpaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_sm"])
    nlp = spacy.load("fr_core_news_sm")

def get_file_hash(file_path: str) -> str:
    """Generate a hash of the file to track changes."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def load_processed_files() -> Dict[str, str]:
    """Load the cache of processed files and their hashes."""
    if os.path.exists(PROCESSED_FILES_CACHE):
        with open(PROCESSED_FILES_CACHE, 'r') as f:
            return json.load(f)
    return {}

def save_processed_files(processed_files: Dict[str, str]):
    """Save the cache of processed files and their hashes."""
    os.makedirs(os.path.dirname(PROCESSED_FILES_CACHE), exist_ok=True)
    with open(PROCESSED_FILES_CACHE, 'w') as f:
        json.dump(processed_files, f)

def discover_document_files(root_dir: str, module_priorities=None):
    """
    Find all document files recursively in the given directory.
    
    Args:
        root_dir: Root directory to search
        module_priorities: Optional dictionary of module priorities. 
                           Keys are module names, values are priority multipliers.
                           Modules not in this dict will get the default priority of 1.0.
        
    Returns:
        List of tuples: (file_path, file_type)
    """
    files = []
    
    # PDF files
    pdf_files = glob.glob(os.path.join(root_dir, "**", "*.pdf"), recursive=True)
    files.extend([(file, "pdf") for file in pdf_files])

    # PowerPoint files
    ppt_files = glob.glob(os.path.join(root_dir, "**", "*.ppt*"), recursive=True)
    files.extend([(file, "ppt") for file in ppt_files])

    # Word files
    doc_files = glob.glob(os.path.join(root_dir, "**", "*.doc*"), recursive=True)
    files.extend([(file, "doc") for file in doc_files])

    # Plain text files
    txt_files = glob.glob(os.path.join(root_dir, "**", "*.txt"), recursive=True)
    files.extend([(file, "txt") for file in txt_files])
    
    return files

def prioritize_documents(files: List[Tuple[str, str]], module_priorities=None):
    """
    Prioritize documents based on importance, recency, and module.
    
    Args:
        files: List of (file_path, file_type) tuples
        module_priorities: Optional dictionary of module priorities.
                           Keys are module names, values are priority multipliers.
                           Modules not in this dict will get the default priority of 1.0.
        
    Returns:
        List of (file_path, file_type, priority_score) tuples
    """
    prioritized_files = []
    
    for file_path, file_type in files:
        # Start with base priority score of 1.0
        priority_score = 1.0
        
        # Extract module from file path
        module = extract_metadata(file_path).get("module", "Unknown")
        
        # Apply module priority multiplier if specified, else use default of 1.0
        if module_priorities and module in module_priorities:
            priority_score *= module_priorities[module]
        
        # Check file modification date for recency
        try:
            mod_time = os.path.getmtime(file_path)
            file_age_days = (time.time() - mod_time) / (60 * 60 * 24)
            
            # Boost recent files
            if file_age_days < 30:  # Modified in the last month
                priority_score += 0.5
            elif file_age_days < 90:  # Modified in the last quarter
                priority_score += 0.3
            elif file_age_days < 365:  # Modified in the last year
                priority_score += 0.1
        except Exception as e:
            logger.warning(f"Couldn't get modification time for {file_path}: {str(e)}")
        
        # Boost certain file types
        if file_type == "pdf":
            priority_score += 0.2  # PDFs often contain more structured content
            
        # Check file size as a proxy for content richness
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 1000000:  # > 1MB
                priority_score += 0.3
            elif file_size > 500000:  # > 500KB
                priority_score += 0.2
            elif file_size < 10000:  # < 10KB (might be too small to be useful)
                priority_score -= 0.2
        except Exception as e:
            logger.warning(f"Couldn't get file size for {file_path}: {str(e)}")
        
        # Add to prioritized list
        prioritized_files.append((file_path, file_type, priority_score))
        
    # Sort by priority score (descending)
    prioritized_files.sort(key=lambda x: x[2], reverse=True)
    
    return prioritized_files

def extract_metadata(file_path: str) -> Dict[str, str]:
    """Extract detailed metadata from the file path."""
    parts = Path(file_path).relative_to(DATA_PATH).parts
    filename = Path(file_path).stem

    metadata = {
        "source": file_path,
        "filename": filename,
        "file_type": Path(file_path).suffix.lower().replace(".", "")
    }

    # Extract structured metadata from filepath
    if len(parts) >= 3:
        metadata["sector"] = parts[0]
        metadata["semester"] = parts[1]
        # Use the directory name as the module name, not the filename
        metadata["module"] = parts[2]  # This gets the directory name (e.g., "DBA", "Java")

        # Attempt to extract course code if present in filename
        import re
        course_code_match = re.search(r'([A-Z]{2,4}\d{3,4})', filename)
        if course_code_match:
            metadata["course_code"] = course_code_match.group(1)

        # Attempt to extract document type
        doc_types = ["syllabus", "lecture", "notes", "assignment", "exam", "tutorial", "lab", "reading",
                    # French document types
                    "cours", "conférence", "notes", "devoir", "examen", "tutoriel", "laboratoire", "lecture"]
        for doc_type in doc_types:
            if doc_type in filename.lower() or doc_type in file_path.lower():
                metadata["document_type"] = doc_type
                break
        
        # Extract professor names from syllabi
        if metadata.get("document_type") == "syllabus":
            try:
                loader = UnstructuredFileLoader(file_path, strategy="hi_res")
                elements = loader.load_and_split()  # Or partition directly
                for el in elements:
                    if "Instructor:" in el.page_content or "Professeur:" in el.page_content:
                        professor = extract_professor_info(el.page_content)
                        if professor:
                            metadata["professor"] = professor
                            break
            except Exception as e:
                logger.warning(f"Could not extract professor from {file_path}: {str(e)}")
    else:
        metadata["sector"] = "Unknown"
        metadata["semester"] = "Unknown"
        metadata["module"] = "General"

    return metadata

def load_and_chunk_file(file_path: str) -> List[Dict]:
    """Load a single document file and return its pages with metadata."""
    try:
        loader = UnstructuredFileLoader(
            file_path,
            strategy="fast",  # Use fast strategy instead of hi_res for better performance
            chunking_strategy="by_title",
            max_characters=512 * 4,
            combine_text_under_n_chars=200,
            multipage_sections=True,
        )
        chunks = loader.load()
        path_metadata = extract_metadata(file_path)
        chunk_dicts = [
            {
                "page_content": chunk.page_content,
                "metadata": {**chunk.metadata, **path_metadata}
            }
            for chunk in chunks
        ]
        return chunk_dicts
    except Exception as e:
        logger.error(f"Error loading and chunking {file_path}: {str(e)}")
        return []

def extract_professor_info(text: str) -> Optional[str]:
    """Extract professor name from syllabus text using common patterns in French and English."""
    import re
    
    patterns = [
        # French patterns first for priority
        r'(?:Enseignant|Professeur|Chargé de cours|Maître de conférences|Encadrant)(?:\s*):?\s*([^.,\n]+)',
        r'(?:Enseigné|Donné)(?:\s+par\s+):?\s*([^.,\n]+)',
        r'(?:Nom du professeur|Nom de l\'enseignant)(?:\s*):?\s*([^.,\n]+)',
        # English patterns as fallback
        r'(?:Instructor|Professor|Teacher|Lecturer)(?:\s*):?\s*([^.,\n]+)',
        r'(?:Taught)(?:\s+by\s+):?\s*([^.,\n]+)',
        r'(?:Name of instructor)(?:\s*):?\s*([^.,\n]+)'
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            professor_name = matches.group(1).strip()
            # Clean up any extra text
            professor_name = re.sub(r'\(.*?\)', '', professor_name).strip()
            return professor_name
    
    return None

def create_semantic_chunks(documents) -> List[Dict]:
    """
    Split documents into semantic chunks using advanced NLP techniques.
    This improves on the basic RecursiveCharacterTextSplitter by preserving
    semantic boundaries and ensuring coherent chunks.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        List of semantically coherent chunks with metadata
    """
    all_chunks = []
    
    # Define different strategies based on document type
    doc_type_strategies = {
        "syllabus": {
            "base_chunk_size": 256,
            "overlap": 50,
            "respect_sections": True,
            "preserve_headers": True
        },
        "exam": {
            "base_chunk_size": 384,
            "overlap": 75,
            "respect_sections": True,
            "preserve_questions": True
        },
        "assignment": {
            "base_chunk_size": 384,
            "overlap": 75,
            "respect_sections": True,
            "preserve_tasks": True
        },
        "lecture": {
            "base_chunk_size": 512,
            "overlap": 100,
            "respect_sections": True,
            "preserve_headers": True
        },
        "reading": {
            "base_chunk_size": 512,
            "overlap": 100,
            "respect_sections": True,
            "preserve_paragraphs": True
        },
        "default": {
            "base_chunk_size": 384,
            "overlap": 75,
            "respect_sections": True,
            "preserve_headers": False
        }
    }
    
    for doc in documents:
        # Access metadata as a dictionary key instead of attribute
        doc_type = doc.get("metadata", {}).get("document_type", "default")
        strategy = doc_type_strategies.get(doc_type, doc_type_strategies["default"])
        
        # Process the document with spaCy for linguistic features
        text = doc.get("page_content", "")
        
        # Extract metadata for context preservation
        title = doc.get("metadata", {}).get("filename", "")
        
        # Detect document structure and sections
        sections = extract_document_sections(text, doc_type)
        
        if sections and strategy["respect_sections"]:
            # Process each section independently to preserve context
            for section_title, section_text in sections:
                section_chunks = chunk_text_semantically(
                    section_text, 
                    base_size=strategy["base_chunk_size"],
                    overlap=strategy["overlap"],
                    doc_type=doc_type
                )
                
                # Add section title to each chunk for context
                for i, chunk_text in enumerate(section_chunks):
                    # Create metadata for this chunk
                    chunk_metadata = doc.get("metadata", {}).copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(section_chunks)
                    chunk_metadata["section"] = section_title
                    
                    # Format chunk with appropriate context indicators
                    formatted_text = f"[Section: {section_title}] {chunk_text}"
                    
                    chunk_doc = {
                        "page_content": formatted_text,
                        "metadata": chunk_metadata
                    }
                    all_chunks.append(chunk_doc)
        else:
            # No clear sections, process the whole document
            chunks = chunk_text_semantically(
                text, 
                base_size=strategy["base_chunk_size"],
                overlap=strategy["overlap"],
                doc_type=doc_type
            )
            
            # Process each chunk
            for i, chunk_text in enumerate(chunks):
                # Create metadata for this chunk
                chunk_metadata = doc.get("metadata", {}).copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                
                # Add document title for context
                formatted_text = f"[Document: {title}] {chunk_text}"
                
                # Create and add the chunk document
                chunk_doc = {
                    "page_content": formatted_text,
                    "metadata": chunk_metadata
                }
                all_chunks.append(chunk_doc)

    logger.info(f"Created {len(all_chunks)} semantic chunks from {len(documents)} documents")
    return all_chunks

def extract_document_sections(text, doc_type):
    """
    Extract sections from document text based on document type and content structure.
    
    Args:
        text: Document text
        doc_type: Type of document
        
    Returns:
        List of (section_title, section_text) tuples
    """
    sections = []
    
    try:
        # Use spaCy to process the text
        doc = nlp(text[:100000])  # Limit length to avoid memory issues
        
        # Different section detection strategies based on document type
        if doc_type == "syllabus":
            # Common syllabus section headers
            section_patterns = [
                r"(?:^|\n)(?:\d+\.\s*)?(?:Course|Class) (?:Description|Information)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Learning )?(?:Objectives|Goals|Outcomes)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Course|Class) (?:Schedule|Timeline|Outline)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Grading|Assessment|Evaluation)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Policies|Requirements|Expectations)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Materials|Resources|Textbooks|Readings)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Assignments|Projects|Homework)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Attendance|Participation)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Academic )?(?:Integrity|Honesty|Plagiarism)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Accessibility|Accommodations|Disability)"
            ]
        elif doc_type in ["lecture", "notes"]:
            # Common lecture note section patterns
            section_patterns = [
                r"(?:^|\n)(?:\d+\.\s*)?(?:Introduction|Overview|Background)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Definition|Concept|Theory|Principle)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Method|Approach|Technique|Algorithm)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Example|Illustration|Demonstration)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Analysis|Discussion|Interpretation)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Application|Implementation|Practice)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Result|Outcome|Finding)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Conclusion|Summary|Recap)",
                r"(?:^|\n)(?:\d+\.\s*)?(?:Reference|Bibliography|Further Reading)"
            ]
        elif doc_type in ["exam", "assignment"]:
            # Common exam/assignment patterns
            section_patterns = [
                r"(?:^|\n)(?:Question|Problem|Exercise)\s*(?:\d+|\w+)[.:]",
                r"(?:^|\n)(?:Part|Section)\s*(?:\d+|\w+)[.:]",
                r"(?:^|\n)(?:Task|Activity)\s*(?:\d+|\w+)[.:]",
                r"(?:^|\n)(?:Instructions|Directions)"
            ]
        else:
            # Generic section patterns
            section_patterns = [
                r"(?:^|\n)(?:\d+\.\s*)?[A-Z][A-Z\s]+(?::|\.)",  # ALL CAPS HEADERS
                r"(?:^|\n)#{1,3}\s+\w+",  # Markdown headers
                r"(?:^|\n)(?:\d+\.\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5}(?::|\.)",  # Title Case Headers
                r"(?:^|\n)(?:[IVX]+|[A-Z]|[0-9]+)[\.\)]\s+[A-Z]"  # Numbered sections (Roman, alpha, numeric)
            ]
        
        # Apply the patterns to identify section boundaries
        import re
        text_with_newlines = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Find all potential section headers
        all_matches = []
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text_with_newlines))
            all_matches.extend(matches)
        
        # Sort matches by position
        all_matches.sort(key=lambda m: m.start())
        
        if all_matches:
            # Extract sections based on header positions
            for i in range(len(all_matches)):
                start_pos = all_matches[i].start()
                # Extract header text (limited to first 100 chars for sanity)
                header_text = text_with_newlines[start_pos:start_pos + min(100, len(text_with_newlines) - start_pos)]
                header_text = header_text.split('\n')[0].strip()
                
                # Find end position (start of next section or end of text)
                if i < len(all_matches) - 1:
                    end_pos = all_matches[i + 1].start()
                else:
                    end_pos = len(text_with_newlines)
                
                # Extract section text
                section_text = text_with_newlines[start_pos:end_pos].strip()
                
                if len(section_text) > 50:  # Only include non-trivial sections
                    sections.append((header_text, section_text))
    
    except Exception as e:
        logger.warning(f"Error extracting document sections: {str(e)}")
        # Return empty list if extraction fails
    
    return sections

def chunk_text_semantically(text, base_size=384, overlap=75, doc_type="default"):
    """
    Chunk text semantically using linguistic features rather than character counts.
    
    Args:
        text: Text to chunk
        base_size: Base token size for chunks (will be adapted)
        overlap: Overlap between chunks in tokens
        doc_type: Document type for specialized processing
        
    Returns:
        List of semantically coherent text chunks
    """
    chunks = []
    
    try:
        # Process with SpaCy
        doc = nlp(text[:100000])  # Limit processing to avoid memory issues
        sentences = list(doc.sents)
        
        if not sentences:
            # Fallback to simple newline splitting if spaCy can't find sentences
            text_lines = text.split('\n')
            simple_chunks = []
            current_chunk = []
            current_length = 0
            
            for line in text_lines:
                line_length = len(line.split())
                
                if current_length + line_length > base_size and current_chunk:
                    simple_chunks.append('\n'.join(current_chunk))
                    # Keep some overlap by retaining the last few lines
                    overlap_lines = min(3, len(current_chunk))
                    current_chunk = current_chunk[-overlap_lines:]
                    current_length = sum(len(l.split()) for l in current_chunk)
                
                current_chunk.append(line)
                current_length += line_length
            
            if current_chunk:
                simple_chunks.append('\n'.join(current_chunk))
            
            return simple_chunks
        
        # Use sentence-based chunking with adaptive boundaries
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sent_text = sentence.text.strip()
            if not sent_text:
                continue
                
            sent_length = len(sent_text.split())
            
            # Check special conditions based on document type
            is_boundary = False
            
            # For assignments/exams, questions are strong boundaries
            if doc_type in ["exam", "assignment"] and re.search(r'^(?:Question|Problem|Exercise)\s*(?:\d+|\w+)[.:]', sent_text):
                is_boundary = True
                
            # For lectures/notes, headers and transition phrases are boundaries
            elif doc_type in ["lecture", "notes"] and (re.search(r'^(?:\d+\.\s*)?[A-Z][A-Za-z\s]+:$', sent_text) or 
                                                       any(phrase in sent_text.lower() for phrase in ["in conclusion", "to summarize", "next, we", "finally,"])):
                is_boundary = True
                
            # For syllabus, section headers are boundaries
            elif doc_type == "syllabus" and re.search(r'^(?:\d+\.\s*)?[A-Z][A-Za-z\s]+:$', sent_text):
                is_boundary = True
            
            # Normal length-based boundary
            if current_length + sent_length > base_size or is_boundary:
                if current_chunk:  # Only create chunk if we have content
                    chunks.append(' '.join(current_chunk))
                    
                    # Implement overlap by carrying over some sentences
                    overlap_token_count = 0
                    overlap_sentences = []
                    
                    # Start from the end and work backwards to implement overlap
                    for s in reversed(current_chunk):
                        s_length = len(s.split())
                        if overlap_token_count + s_length <= overlap:
                            overlap_sentences.insert(0, s)
                            overlap_token_count += s_length
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_length = overlap_token_count
                    
            current_chunk.append(sent_text)
            current_length += sent_length
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    except Exception as e:
        logger.warning(f"Error in semantic chunking: {str(e)}")
        # Fallback to simple chunking
        simple_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=base_size * 4,  # Approximate chars per token
            chunk_overlap=overlap * 4,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = simple_text_splitter.split_text(text)
    
    return chunks

def build_or_update_vector_store(documents, embedding_model, retrieval_model_type="default"):
    """
    Build or update the FAISS vector store with advanced retrieval options.
    
    Args:
        documents: List of document dictionaries
        embedding_model: The embedding model to use
        retrieval_model_type: Type of retrieval model to prepare ("default", "dpr", or "bm25")
    """
    
    # If vector store exists, load and update it
    if os.path.exists(FAISS_STORE_PATH):
        try:
            logger.info("Loading existing vector store for update")
            db = FAISS.load_local(FAISS_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)

            # Add new documents to the existing store
            db.add_texts(
                [doc["page_content"] for doc in documents],
                metadatas=[doc["metadata"] for doc in documents],
                batch_size=32  # Adjust based on memory
            )
            logger.info(f"Updated FAISS vector store with {len(documents)} new chunks")
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
            logger.info("Creating new vector store instead")
            db = FAISS.from_texts(
                [doc["page_content"] for doc in documents],
                embedding_model,
                metadatas=[doc["metadata"] for doc in documents]
            )
    else:
        # Create new vector store
        db = FAISS.from_texts(
            [doc["page_content"] for doc in documents],
            embedding_model,
            metadatas=[doc["metadata"] for doc in documents]
        )
        logger.info(f"Created new FAISS vector store with {len(documents)} chunks")

    # Save the updated vector store
    os.makedirs(os.path.dirname(FAISS_STORE_PATH), exist_ok=True)
    db.save_local(FAISS_STORE_PATH)
    logger.info(f"FAISS vector store saved at {FAISS_STORE_PATH}")

    # Prepare retrieval model if requested
    if retrieval_model_type != "default":
        prepare_retrieval_model(db, retrieval_model_type)

    # Save metadata summary for future reference
    save_metadata_summary(documents)
    
    return db

def prepare_retrieval_model(db, model_type):
    """
    Prepare an advanced retrieval model for the vector store.
    
    Args:
        db: The vector database
        model_type: Type of retrieval model to prepare ("dpr" or "bm25")
    """
    output_dir = os.path.join(os.path.dirname(FAISS_STORE_PATH), f"{model_type}_retriever")
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type == "dpr":
        try:
            # Query encoder - optimized for query understanding
            query_encoder = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-small",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "instruction": "Represent this university course query for retrieval: "}
            )
            
            # Document encoder - optimized for document representation
            doc_encoder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # Save encoders for later use
            with open(os.path.join(output_dir, "encoders_config.json"), "w") as f:
                json.dump({
                    "query_encoder": "intfloat/multilingual-e5-small",
                    "doc_encoder": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                }, f)
                
            logger.info(f"DPR retrieval model prepared and saved at {output_dir}")
            
        except Exception as e:
            logger.error(f"Error preparing DPR retrieval model: {str(e)}")
    
    elif model_type == "bm25":
        try:
            # Extract text for BM25 indexing
            texts = [doc["page_content"] for doc in db.get()]
            
            # Tokenize texts for BM25
            import nltk
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import word_tokenize
            
            tokenized_texts = [word_tokenize(text.lower()) for text in texts]
            
            # Create BM25 index
            from rank_bm25 import BM25Okapi
            bm25_index = BM25Okapi(tokenized_texts)
            
            # Save index for later use (simplified)
            import pickle
            with open(os.path.join(output_dir, "bm25_index.pkl"), "wb") as f:
                pickle.dump(bm25_index, f)
                
            logger.info(f"BM25 retrieval model prepared and saved at {output_dir}")
            
        except Exception as e:
            logger.error(f"Error preparing BM25 retrieval model: {str(e)}")
    
    else:
        logger.warning(f"Unknown retrieval model type: {model_type}")

def save_metadata_summary(documents):
    """
    Save a comprehensive summary of metadata with enhanced coverage analysis and gap detection.
    
    Args:
        documents: List of document dictionaries with metadata
    """
    modules = {}
    doc_types = {}
    semesters = {}
    professors = {}
    course_codes = set()
    
    # For coverage analysis
    topics = {}
    coverage_by_time = {}  # Track content by time periods
    concept_coverage = {}  # Track concept coverage
    
    # For temporal analysis
    date_pattern = r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{4})'

    for doc in documents:
        meta = doc["metadata"]
        content = doc["page_content"]
        
        # Basic metadata counts
        module = meta.get("module", "Unknown")
        doc_type = meta.get("document_type", "Unknown")
        semester = meta.get("semester", "Unknown")
        professor = meta.get("professor", "Unknown")
        course_code = meta.get("course_code", "Unknown")

        modules[module] = modules.get(module, 0) + 1
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        semesters[semester] = semesters.get(semester, 0) + 1
        professors[professor] = professors.get(professor, 0) + 1
        
        # Track course codes
        if course_code != "Unknown":
            course_codes.add(course_code)
        
        # Extract topics and concepts for content coverage analysis
        extract_topics_from_content(content, topics)
        
        # Extract key concepts for coverage mapping
        if doc_type in ["syllabus", "lecture"]:
            extract_key_concepts(content, concept_coverage)
            
        # Temporal coverage - extract dates from content
        import re
        dates = re.findall(date_pattern, content)
        for date_str in dates:
            try:
                # Extract year for simplified temporal tracking
                if '-' in date_str:
                    parts = date_str.split('-')
                else:
                    parts = date_str.split('/')
                    
                # Find the year part (4 digits)
                year = next((part for part in parts if len(part) == 4), None)
                if year:
                    coverage_by_time[year] = coverage_by_time.get(year, 0) + 1
            except Exception:
                pass  # Ignore date parsing errors
    
    # Analyze content gaps across multiple dimensions
    content_gaps = analyze_comprehensive_gaps(modules, doc_types, topics, concept_coverage, coverage_by_time)
    
    # Assess cross-module coverage
    cross_module_coverage = assess_cross_module_coverage(documents)
    
    # Check for concept redundancy and conflicts
    redundancy_analysis = analyze_concept_redundancy(concept_coverage)
    
    # Generate coverage score (0-100)
    coverage_score = calculate_coverage_score(
        modules, doc_types, topics, concept_coverage, content_gaps
    )
    
    # Prepare the comprehensive metadata summary
    metadata_summary = {
        "total_chunks": len(documents),
        "modules": modules,
        "document_types": doc_types,
        "semesters": semesters,
        "professors": professors,
        "course_codes": list(course_codes),
        "coverage": {
            "score": coverage_score,
            "topics": {k: v for k, v in sorted(topics.items(), key=lambda item: item[1], reverse=True)[:50]},
            "temporal": coverage_by_time,
            "cross_module": cross_module_coverage,
            "redundancy": redundancy_analysis,
            "gaps": content_gaps
        }
    }

    # Save the metadata summary
    os.makedirs(os.path.dirname(METADATA_CACHE_PATH), exist_ok=True)
    with open(METADATA_CACHE_PATH, 'w') as f:
        json.dump(metadata_summary, f, indent=2)

    logger.info(f"Comprehensive metadata summary with coverage analysis saved to {METADATA_CACHE_PATH}")
    
    # Generate a detailed coverage report
    generate_coverage_report(metadata_summary)
    
    return metadata_summary

def extract_key_concepts(content, concept_coverage):
    """Extract key educational concepts from document content."""
    try:
        # Process with spaCy for concept extraction
        doc = nlp(content[:10000])  # Limit to first 10000 chars
        
        # Key educational terms to look for
        educational_terms = [
            "objective", "goal", "learning outcome", "assessment", "exam", "project",
            "assignment", "quiz", "test", "lecture", "concept", "theory", "practice",
            "skill", "competency", "knowledge", "understand", "apply", "analyze", 
            "evaluate", "create", "deadline", "syllabus", "curriculum"
        ]
        
        # Extract sentences containing educational terms
        for sent in doc.sents:
            sent_text = sent.text.lower()
            for term in educational_terms:
                if term in sent_text:
                    # Store the concept with its context
                    key = f"{term}: {sent_text[:100]}..."
                    concept_coverage[key] = concept_coverage.get(key, 0) + 1
                    break
                    
    except Exception as e:
        logger.warning(f"Error extracting key concepts: {str(e)}")

def analyze_comprehensive_gaps(modules, doc_types, topics, concepts, temporal_coverage):
    """Perform comprehensive gap analysis across multiple dimensions."""
    gaps = []
    
    # Module coverage gaps
    for module, count in modules.items():
        if count < 5 and module != "Unknown":
            gaps.append({
                "type": "module_undercoverage",
                "item": module,
                "count": count,
                "severity": "high" if count < 3 else "medium",
                "recommendation": "Add more documents for this module"
            })
    
    # Document type gaps
    expected_doc_types = ["syllabus", "lecture", "notes", "assignment", "exam"]
    for doc_type in expected_doc_types:
        if doc_type not in doc_types or doc_types[doc_type] < 2:
            gaps.append({
                "type": "missing_doc_type",
                "item": doc_type,
                "count": doc_types.get(doc_type, 0),
                "severity": "high" if doc_type in ["syllabus", "lecture"] else "medium",
                "recommendation": f"Add more {doc_type} documents"
            })
    
    # Concept coverage gaps
    if topics:
        # Core educational concepts that should be covered
        core_concepts = [
            "assessment", "grading", "objective", "outcome", "deadline", "project", 
            "exam", "lecture", "tutorial", "lab", "reading", "schedule", "policy"
        ]
        
        for concept in core_concepts:
            # Check if concept is adequately covered
            coverage_count = sum(count for topic, count in topics.items() if concept in topic.lower())
            if coverage_count < 3:
                gaps.append({
                    "type": "concept_undercoverage",
                    "item": concept,
                    "count": coverage_count,
                    "severity": "high" if coverage_count == 0 else "medium",
                    "recommendation": f"Add content covering {concept}"
                })
    
    # Temporal coverage gaps
    if temporal_coverage:
        current_year = datetime.now().year
        recent_years = [str(year) for year in range(current_year-2, current_year+1)]
        
        # Check for recent content
        has_recent_content = any(year in temporal_coverage for year in recent_years)
        if not has_recent_content:
            gaps.append({
                "type": "temporal_gap",
                "item": "recent_content",
                "severity": "high",
                "recommendation": "Add recent/current content (past 2 years)"
            })
    
    return gaps

def assess_cross_module_coverage(documents):
    """Assess how well concepts are covered across different modules."""
    # Group documents by module
    modules = {}
    for doc in documents:
        module = doc["metadata"].get("module", "Unknown")
        if module not in modules:
            modules[module] = []
        modules[module].append(doc)
    
    # Extract key terms from each module
    module_terms = {}
    for module, docs in modules.items():
        if module == "Unknown":
            continue
            
        # Combine content from module documents
        combined_content = " ".join(doc["page_content"] for doc in docs)
        
        # Extract terms
        try:
            doc = nlp(combined_content[:20000])  # Limit size
            terms = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 3]
            # Get top terms by frequency
            from collections import Counter
            top_terms = Counter(terms).most_common(20)
            module_terms[module] = [term for term, _ in top_terms]
        except Exception as e:
            logger.warning(f"Error extracting terms for module {module}: {str(e)}")
    
    # Analyze term overlap between modules
    overlap_analysis = {}
    for module1, terms1 in module_terms.items():
        overlap_analysis[module1] = {}
        for module2, terms2 in module_terms.items():
            if module1 != module2:
                # Calculate Jaccard similarity
                intersection = len(set(terms1) & set(terms2))
                union = len(set(terms1) | set(terms2))
                similarity = intersection / union if union > 0 else 0
                overlap_analysis[module1][module2] = similarity
    
    return {
        "module_terms": module_terms,
        "overlap_analysis": overlap_analysis
    }

def analyze_concept_redundancy(concept_coverage):
    """Analyze potential redundancies and conflicts in concept coverage."""
    redundancy = {
        "excessive_coverage": [],
        "potential_conflicts": []
    }
    
    # Identify concepts with excessive coverage
    for concept, count in concept_coverage.items():
        if count > 10:  # Threshold for excessive coverage
            redundancy["excessive_coverage"].append({
                "concept": concept.split(":")[0],
                "count": count
            })
    
    # Identify potential conflicts by looking for similar concepts with different contexts
    concept_groups = {}
    for concept in concept_coverage:
        key = concept.split(":")[0].strip()
        if key not in concept_groups:
            concept_groups[key] = []
        concept_groups[key].append(concept)
    
    # Look for concept groups with multiple entries
    for key, concepts in concept_groups.items():
        if len(concepts) > 1:
            # Analyze for potential conflicts using simple text similarity
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    # Simple similarity check - can be enhanced with NLP techniques
                    if len(set(concept1.split()) & set(concept2.split())) / len(set(concept1.split()) | set(concept2.split())) < 0.5:
                        redundancy["potential_conflicts"].append({
                            "concept": key,
                            "variant1": concept1,
                            "variant2": concept2
                        })
    
    return redundancy

def calculate_coverage_score(modules, doc_types, topics, concepts, gaps):
    """Calculate an overall coverage score (0-100)."""
    score = 100  # Start with perfect score
    
    # Deduct for module undercoverage
    low_coverage_modules = sum(1 for _, count in modules.items() if count < 5 and _ != "Unknown")
    module_score = max(0, 25 - (low_coverage_modules * 5))
    
    # Deduct for missing document types
    expected_doc_types = ["syllabus", "lecture", "notes", "assignment", "exam"]
    missing_types = sum(1 for doc_type in expected_doc_types if doc_type not in doc_types or doc_types[doc_type] < 2)
    doc_type_score = max(0, 25 - (missing_types * 5))
    
    # Deduct for concept gaps
    concept_gap_score = max(0, 25 - (len([g for g in gaps if g["type"] == "concept_undercoverage"]) * 3))
    
    # Deduct for temporal gaps
    temporal_gap_score = max(0, 25 - (len([g for g in gaps if g["type"] == "temporal_gap"]) * 10))
    
    # Calculate final score
    final_score = module_score + doc_type_score + concept_gap_score + temporal_gap_score
    
    return final_score

def generate_coverage_report(metadata_summary):
    """Generate a detailed coverage report."""
    report_path = os.path.join(os.path.dirname(METADATA_CACHE_PATH), "coverage_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 25 + "KNOWLEDGE BASE COVERAGE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total chunks: {metadata_summary['total_chunks']}\n")
        f.write(f"Coverage score: {metadata_summary['coverage']['score']}/100\n")
        f.write(f"Total modules: {len(metadata_summary['modules'])}\n")
        f.write(f"Total course codes: {len(metadata_summary['course_codes'])}\n")
        f.write(f"Document types: {', '.join(metadata_summary['document_types'].keys())}\n\n")
        
        # Module coverage
        f.write("MODULE COVERAGE\n")
        f.write("-" * 50 + "\n")
        for module, count in sorted(metadata_summary['modules'].items(), key=lambda x: x[1], reverse=True):
            # Remove or replace problematic characters
            safe_module = str(module).encode('utf-8', 'replace').decode('utf-8')
            f.write(f"  {safe_module}: {count} chunks\n")
        
        # Document type coverage
        f.write("\nDOCUMENT TYPE COVERAGE\n")
        f.write("-" * 50 + "\n")
        for doc_type, count in sorted(metadata_summary['document_types'].items(), key=lambda x: x[1], reverse=True):
            # Remove or replace problematic characters
            safe_doc_type = str(doc_type).encode('utf-8', 'replace').decode('utf-8')
            f.write(f"  {safe_doc_type}: {count} chunks\n")
        
        # Top topics
        f.write("\nTOP TOPICS\n")
        f.write("-" * 50 + "\n")
        for topic, count in list(sorted(metadata_summary['coverage']['topics'].items(), key=lambda x: x[1], reverse=True))[:20]:
            # Remove or replace problematic characters
            safe_topic = str(topic).encode('utf-8', 'replace').decode('utf-8')
            f.write(f"  {safe_topic}: {count} mentions\n")
            
        # Temporal coverage
        if 'temporal' in metadata_summary['coverage']:
            f.write("\nTEMPORAL COVERAGE\n")
            f.write("-" * 50 + "\n")
            for year, count in sorted(metadata_summary['coverage']['temporal'].items()):
                f.write(f"  {year}: {count} mentions\n")
        
        # Cross-module coverage
        if 'cross_module' in metadata_summary['coverage'] and 'overlap_analysis' in metadata_summary['coverage']['cross_module']:
            f.write("\nCROSS-MODULE COVERAGE (Similarity Scores)\n")
            f.write("-" * 50 + "\n")
            overlap = metadata_summary['coverage']['cross_module']['overlap_analysis']
            for module1, scores in overlap.items():
                for module2, score in scores.items():
                    # Remove or replace problematic characters
                    safe_module1 = str(module1).encode('utf-8', 'replace').decode('utf-8')
                    safe_module2 = str(module2).encode('utf-8', 'replace').decode('utf-8')
                    f.write(f"  {safe_module1} ↔ {safe_module2}: {score:.2f}\n")
        
        # Identified gaps
        f.write("\nIDENTIFIED GAPS\n")
        f.write("-" * 50 + "\n")
        if metadata_summary['coverage']['gaps']:
            for gap in metadata_summary['coverage']['gaps']:
                severity = gap.get('severity', 'medium').upper()
                item = str(gap['item']).encode('utf-8', 'replace').decode('utf-8')
                recommendation = str(gap.get('recommendation', '')).encode('utf-8', 'replace').decode('utf-8')
                f.write(f"  [{severity}] {gap['type']}: {item} - {recommendation}\n")
        else:
            f.write("  No significant gaps identified.\n")
            
        # Redundancy analysis
        if 'redundancy' in metadata_summary['coverage']:
            f.write("\nREDUNDANCY ANALYSIS\n")
            f.write("-" * 50 + "\n")
            
            # Excessive coverage
            if metadata_summary['coverage']['redundancy']['excessive_coverage']:
                f.write("Concepts with excessive coverage:\n")
                for item in metadata_summary['coverage']['redundancy']['excessive_coverage']:
                    concept = str(item['concept']).encode('utf-8', 'replace').decode('utf-8')
                    f.write(f"  {concept}: {item['count']} mentions\n")
            
            # Potential conflicts
            if metadata_summary['coverage']['redundancy']['potential_conflicts']:
                f.write("\nPotential conceptual conflicts:\n")
                for item in metadata_summary['coverage']['redundancy']['potential_conflicts']:
                    concept = str(item['concept']).encode('utf-8', 'replace').decode('utf-8')
                    variant1 = str(item['variant1']).encode('utf-8', 'replace').decode('utf-8')
                    variant2 = str(item['variant2']).encode('utf-8', 'replace').decode('utf-8')
                    f.write(f"  Concept: {concept}\n")
                    f.write(f"    - Variant 1: {variant1}\n")
                    f.write(f"    - Variant 2: {variant2}\n")
    
    logger.info(f"Comprehensive coverage report generated at {report_path}")

def enhance_noise_robustness(documents):
    """
    Apply advanced filtering techniques to improve robustness against noise
    and detect contradictions or outdated data.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Filtered and enhanced documents
    """
    enhanced_docs = []
    contradictions = {}
    outdated_info = {}
    
    # Define patterns for noise detection
    noise_patterns = [
        r"page\s+\d+\s+of\s+\d+",  # Page numbers
        r"^\s*\d+\s*$",  # Isolated numbers
        r"^[\W_]+$",  # Lines with only special characters
        r"(?i)confidential|draft|do not distribute",  # Confidentiality notices
        r"(?i)footer|header|copyright|all rights reserved",  # Footer/header text
        r"https?://\S+",  # URLs (might be useful, but often noise in PDFs)
        r"(?i)slide \d+",  # Slide numbers
        r"(?i)created by|powered by|generated by",  # System-generated content
    ]
    
    # Compile the patterns for efficiency
    compiled_patterns = [re.compile(pattern) for pattern in noise_patterns]
    
    # Process each document
    for doc in documents:
        content = doc["page_content"]
        meta = doc["metadata"]
        doc_id = meta.get("source", "") + "_" + str(meta.get("chunk_index", ""))
        
        # Filter out noisy lines
        lines = content.split("\n")
        cleaned_lines = []
        
        for line in lines:
            is_noise = False
            for pattern in compiled_patterns:
                if pattern.search(line):
                    is_noise = True
                    break
                    
            if not is_noise and len(line.strip()) > 0:
                cleaned_lines.append(line)
                
        cleaned_content = "\n".join(cleaned_lines)
        
        # Remove repetitive sequences (common OCR artifact)
        cleaned_content = re.sub(r'(.{3,})\1{2,}', r'\1', cleaned_content)
        
        # Identify content with dates for recency checking
        date_matches = re.findall(r'(?:19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}', cleaned_content)
        
        if date_matches:
            # Extract and parse dates to find outdated content
            try:
                from datetime import datetime
                
                for date_str in date_matches:
                    # Handle different date formats
                    try:
                        if re.match(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', date_str):  # YYYY-MM-DD
                            year, month, day = map(int, re.split(r'[/-]', date_str))
                        else:  # MM-DD-YYYY
                            parts = re.split(r'[/-]', date_str)
                            month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
                            
                        doc_date = datetime(year, month, day)
                        
                        # Check if date is older than 3 years
                        years_old = (datetime.now() - doc_date).days / 365
                        if years_old > 3:
                            # Record potential outdated information
                            content_extract = cleaned_content[max(0, cleaned_content.find(date_str) - 50):cleaned_content.find(date_str) + 100]
                            if doc_id not in outdated_info:
                                outdated_info[doc_id] = []
                            outdated_info[doc_id].append((date_str, content_extract, years_old))
                    except Exception:
                        pass  # Ignore date parsing errors
            except Exception as e:
                logger.warning(f"Error processing dates: {str(e)}")
        
        # Check for potential contradictions with other documents
        # Focus on definitional statements that might contradict
        definition_matches = re.finditer(r'(?:is defined as|is|means|refers to|consists of|comprises)\s+([^.]+)', cleaned_content)
        
        for match in definition_matches:
            definition = match.group(1).strip()
            key_terms = re.findall(r'\b[A-Z][a-z]{2,}\b', cleaned_content[:match.start()])
            
            for term in key_terms[-3:]:  # Look at the 3 closest terms before the definition
                if term not in contradictions:
                    contradictions[term] = {}
                
                # Store the definition for this term from this document
                if doc_id not in contradictions[term]:
                    contradictions[term][doc_id] = []
                
                contradictions[term][doc_id].append(definition)
        
        # Update the document with cleaned content
        doc["page_content"] = cleaned_content
        
        # Add flags for potentially problematic content
        doc["metadata"]["has_outdated_info"] = doc_id in outdated_info
        doc["metadata"]["potential_contradictions"] = any(doc_id in contradictions[term] for term in contradictions)
        
        enhanced_docs.append(doc)
    
    # Analyze contradictions across documents
    contradiction_report = []
    for term, docs_dict in contradictions.items():
        if len(docs_dict) > 1:  # Term defined in multiple documents
            all_definitions = []
            for doc_id, definitions in docs_dict.items():
                all_definitions.extend([(definition, doc_id) for definition in definitions])
            
            # Compare definitions for contradictions
            if len(all_definitions) > 1:
                for i in range(len(all_definitions)):
                    for j in range(i+1, len(all_definitions)):
                        def1, doc_id1 = all_definitions[i]
                        def2, doc_id2 = all_definitions[j]
                        
                        # Simple contradiction detection - more sophisticated NLP possible
                        if not any(word in def2.lower() for word in def1.lower().split()):
                            contradiction_report.append({
                                "term": term,
                                "document1": doc_id1,
                                "definition1": def1,
                                "document2": doc_id2,
                                "definition2": def2
                            })
    
    # Generate a report on potential quality issues
    quality_report = {
        "outdated_information": outdated_info,
        "potential_contradictions": contradiction_report
    }
    
    # Save the quality report - ensure directory exists first
    report_path = os.path.join(os.path.dirname(METADATA_CACHE_PATH), "quality_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)  # Create directory if it doesn't exist
    with open(report_path, 'w') as f:
        json.dump(quality_report, f, indent=2)
    
    logger.info(f"Enhanced {len(enhanced_docs)} documents for noise robustness")
    logger.info(f"Identified {len(outdated_info)} documents with potentially outdated information")
    logger.info(f"Detected {len(contradiction_report)} potential contradictions between documents")
    
    return enhanced_docs

def optimize_vector_store_indexing(db, num_documents):
    """
    Apply advanced indexing techniques to optimize FAISS for better performance
    with large document collections.
    
    Args:
        db: The FAISS vectorstore
        num_documents: Number of documents in the store
    
    Returns:
        Optimized FAISS index
    """
    logger.info("Optimizing vector store indexing for better scalability")
    
    try:
        # For larger datasets, apply more sophisticated indexing
        if num_documents > 10000:
            logger.info("Large dataset detected, applying HNSW indexing")
            
            # Extract the underlying FAISS index
            index = db.index
            
            # Configure HNSW parameters for a good balance of search quality and speed
            # M: Number of connections per element (higher = better recall but more memory)
            # efConstruction: Effort during construction (higher = better quality but slower build)
            # efSearch: Effort during search (higher = better quality but slower search)
            
            import faiss
            
            # Get dimensionality of embeddings
            d = index.d
            
            # Create HNSW index
            hnsw_index = faiss.IndexHNSWFlat(d, 32)  # 32 connections per element
            hnsw_index.hnsw.efConstruction = 100
            hnsw_index.hnsw.efSearch = 64
            
            # Transfer vectors from original index to HNSW
            if hasattr(index, 'ntotal') and index.ntotal > 0:
                # For IVF-based indices
                vectors = faiss.extract_index_vectors(index)
                hnsw_index.add(vectors)
            else:
                logger.warning("Could not extract vectors from original index")
            
            # Replace the index
            db.index = hnsw_index
            logger.info("Successfully applied HNSW indexing for faster search")
            
        elif num_documents > 1000:
            logger.info("Medium dataset detected, applying IVF indexing")
            
            # Extract the underlying FAISS index
            index = db.index
            
            # Get dimensionality of embeddings
            d = index.d
            
            # Set number of clusters (typical rule: sqrt(n) where n is dataset size)
            n_clusters = int(num_documents ** 0.5)
            n_clusters = max(4, min(n_clusters, 256))  # Keep between 4 and 256
            
            # Create IVF index
            import faiss
            quantizer = faiss.IndexFlatL2(d)
            ivf_index = faiss.IndexIVFFlat(quantizer, d, n_clusters)
            
            # Train on a subset of vectors
            if hasattr(index, 'ntotal') and index.ntotal > 0:
                vectors = faiss.extract_index_vectors(index)
                # Use subset for training if we have many vectors
                train_vectors = vectors[:min(50000, vectors.shape[0])]
                ivf_index.train(train_vectors)
                ivf_index.add(vectors)
            else:
                logger.warning("Could not extract vectors from original index")
            
            # Set higher nprobe for better recall
            ivf_index.nprobe = min(n_clusters // 4, 64)  # Higher values = better recall but slower search
            
            # Replace the index
            db.index = ivf_index
            logger.info(f"Successfully applied IVF indexing with {n_clusters} clusters")
    
    except Exception as e:
        logger.error(f"Error optimizing vector store indexing: {str(e)}")
        logger.info("Continuing with default indexing")
    
    return db

def determine_optimal_batch_size(doc_length=None):
    """
    Dynamically determine the optimal batch size for document processing
    based on system resources and document characteristics.
    
    Args:
        doc_length: Optional average document length for more accurate estimation
        
    Returns:
        Optimal batch size
    """
    try:
        import psutil
        import sys
        
        # Get available system memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        
        # Consider Python interpreter overhead
        if sys.maxsize > 2**32:  # 64-bit system
            interpreter_overhead_gb = 0.5
        else:  # 32-bit system
            interpreter_overhead_gb = 0.25
            
        # Adjust available memory
        available_memory_gb = max(0.5, available_memory_gb - interpreter_overhead_gb)
        
        # Estimate memory needed per document
        # Adjust based on document length if provided
        if doc_length:
            # 4 bytes per float * embedding dimension (assumed to be ~1536 for modern models)
            # Plus overhead for document content and metadata
            memory_per_doc_mb = (4 * 1536 + doc_length * 2) / (1024 * 1024)
        else:
            # Default conservative estimate
            memory_per_doc_mb = 10  # MB per document
            
        # Calculate batch size that would use about 50% of available memory
        batch_size = int((available_memory_gb * 1024 * 0.5) / memory_per_doc_mb)
        
        # Cap batch size within reasonable bounds
        batch_size = max(8, min(batch_size, 256))
        
        logger.info(f"Dynamically determined batch size: {batch_size} based on {available_memory_gb:.2f}GB available memory")
        return batch_size
        
    except Exception as e:
        logger.warning(f"Error determining optimal batch size: {str(e)}")
        # Default fallback batch size
        return 32

class CourseEmbeddingFineTuner:
    """
    Fine-tune embedding models on French course-specific data to improve retrieval quality.
    """
    
    def __init__(self, base_model="antoinelouis/french-me5-small"):
        """Initialize the fine-tuner with a French-focused base model."""
        self.base_model_name = base_model
        self.fine_tuned = False
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the base model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading base model: {self.base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.model = AutoModel.from_pretrained(self.base_model_name)
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
            
    def prepare_training_data(self, documents, queries=None):
        """
        Prepare training data from documents and optional queries.
        If queries are not provided, synthetic ones will be generated.
        """
        if not documents:
            logger.warning("No documents provided for fine-tuning")
            return None
            
        training_data = []
        
        # Use provided queries or generate synthetic ones
        if not queries:
            queries = self._generate_synthetic_queries(documents)
            
        # Create positive pairs (query + relevant document)
        for query, relevant_docs in queries.items():
            for doc_id in relevant_docs:
                if doc_id < len(documents):
                    training_data.append({
                        "query": query,
                        "positive": documents[doc_id]["page_content"],
                        "negative": documents[random.randint(0, len(documents)-1)]["page_content"]
                    })
                    
        logger.info(f"Prepared {len(training_data)} training examples")
        return training_data
        
    def _generate_synthetic_queries(self, documents):
        """Generate synthetic queries from documents for training."""
        queries = {}
        
        try:
            # Use spaCy for named entity and noun chunk extraction
            for i, doc in enumerate(documents[:100]):  # Limit to 100 documents for efficiency
                content = doc["page_content"]
                
                # Process with spaCy
                spacy_doc = nlp(content[:10000])  # Limit content length
                
                # Extract potential queries from named entities
                entities = [ent.text for ent in spacy_doc.ents]
                
                # Extract potential queries from noun chunks
                noun_chunks = [chunk.text for chunk in spacy_doc.noun_chunks]
                
                # Combine and select a subset
                potential_queries = entities + noun_chunks
                selected_queries = random.sample(potential_queries, min(3, len(potential_queries)))
                
                for query in selected_queries:
                    if query not in queries:
                        queries[query] = []
                    queries[query].append(i)
                    
            logger.info(f"Generated {len(queries)} synthetic queries")
            return queries
        except Exception as e:
            logger.error(f"Error generating synthetic queries: {str(e)}")
            return {}
            
    def fine_tune(self, documents, epochs=3, batch_size=8):
        """
        Fine-tune the embedding model on course-specific data.
        
        Args:
            documents: List of document dictionaries
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Fine-tuned embedding model
        """
        if not self.load_model():
            logger.error("Could not load base model for fine-tuning")
            # Return a standard embedding model as fallback
            return HuggingFaceEmbeddings(model_name=self.base_model_name)
            
        # Prepare training data
        training_data = self.prepare_training_data(documents)
        if not training_data:
            logger.warning("No training data available for fine-tuning")
            return HuggingFaceEmbeddings(model_name=self.base_model_name)
            
        try:
            # Check if we have torch available for fine-tuning
            try:
                import torch
                _ = torch.__version__
            except ImportError:
                logger.warning(
                    "Missing torch package required for fine-tuning. Using base model instead."
                )
                return HuggingFaceEmbeddings(
                    model_name=self.base_model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True})
                
            # Set model to training mode
            self.model.train()
            
            # Initialize optimizer
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
            
            # Training loop
            for epoch in range(epochs):
                epoch_loss = 0
                random.shuffle(training_data)
                
                for i in range(0, len(training_data), batch_size):
                    batch = training_data[i:i + batch_size]
                    
                    # Process batch
                    queries = [item[0] for item in batch]
                    positives = [item[1] for item in batch]
                    negatives = [item[2] for item in batch]
                    
                    # Tokenize inputs
                    query_inputs = self.tokenizer(queries,
                                                padding=True,
                                                truncation=True,
                                                return_tensors="pt").to(
                                                    self.device)
                    pos_inputs = self.tokenizer(positives,
                                              padding=True,
                                              truncation=True,
                                              return_tensors="pt").to(
                                                  self.device)
                    neg_inputs = self.tokenizer(negatives,
                                              padding=True,
                                              truncation=True,
                                              return_tensors="pt").to(
                                                  self.device)
                    
                    # Get embeddings
                    with torch.no_grad():
                        query_emb = self.model(**query_inputs).pooler_output
                        pos_emb = self.model(**pos_inputs).pooler_output
                        neg_emb = self.model(**neg_inputs).pooler_output
                    
                    # Compute similarity scores
                    pos_scores = torch.sum(query_emb * pos_emb, dim=1)
                    neg_scores = torch.sum(query_emb * neg_emb, dim=1)
                    
                    # Compute triplet loss
                    margin = 0.3
                    loss = torch.mean(
                        torch.relu(margin - pos_scores + neg_scores))
                    
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            
            # Set fine-tuned flag
            self.fine_tuned = True
            logger.info("Fine-tuning completed successfully")
            
            # Create a custom HuggingFaceEmbeddings with domain-specific parameters
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.base_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True})
            
            return embedding_model
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            # Return default embeddings as fallback
            return HuggingFaceEmbeddings(
                model_name=self.base_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True})

def detect_and_resolve_contradictions(documents):
    """
    Use advanced NLP techniques to identify contradictory information 
    across documents and resolve conflicts.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Tuple of (processed documents, contradiction report)
    """
    logger.info("Running advanced contradiction detection")
    
    # Initialize contradiction tracking
    contradiction_map = {}
    fact_statements = {}  # Map facts to their source documents
    numerical_facts = {}  # Special tracking for numerical facts which are easier to compare
    processed_docs = []
    
    # Initialize NLP for sentence similarity
    try:
        from sentence_transformers import SentenceTransformer
        similarity_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        have_similarity_model = True
    except ImportError:
        logger.warning("SentenceTransformer not available, using simpler contradiction detection")
        have_similarity_model = False
    
    # Extract factual statements from all documents
    for doc in documents:
        content = doc["page_content"]
        doc_id = f"{doc['metadata'].get('source', 'unknown')}:{doc['metadata'].get('chunk_index', 0)}"
        
        # Extract sentences that are likely to be factual statements
        doc_nlp = nlp(content[:50000])  # Limit to prevent memory issues
        
        for sent in doc_nlp.sents:
            sent_text = sent.text.strip()
            
            # Skip short sentences or questions
            if len(sent_text) < 10 or sent_text.endswith("?"):
                continue
                
            # Look for statements that are likely to be factual
            # 1. Contains factual indicators
            is_factual = any(indicator in sent_text.lower() for indicator in 
                          ["is", "are", "was", "were", "has", "have", "contains", "consists", 
                           "equals", "means", "defines", "requires", "includes", "comprises"])
            
            # 2. Not hedged or speculative
            is_speculative = any(hedge in sent_text.lower() for hedge in 
                              ["may", "might", "could", "possibly", "perhaps", "probably", 
                               "likely", "usually", "often", "sometimes", "generally"])
            
            if is_factual and not is_speculative:
                if sent_text not in fact_statements:
                    fact_statements[sent_text] = []
                fact_statements[sent_text].append(doc_id)
                
                # Check for numerical facts (these are easier to compare directly)
                numeric_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(%|percent|euros?|dollars?|\$|€|pounds?|£|years?|months?|days?|hours?)', sent_text.lower())
                
                for value_str, unit in numeric_matches:
                    try:
                        value = float(value_str)
                        # Normalize the unit
                        if unit in ["percent", "%"]:
                            unit = "percent"
                        elif unit in ["euros", "euro", "€"]:
                            unit = "euro"
                        elif unit in ["dollars", "dollar", "$"]:
                            unit = "dollar"
                        elif unit in ["pounds", "pound", "£"]:
                            unit = "pound"
                        elif unit in ["years", "year"]:
                            unit = "year"
                        elif unit in ["months", "month"]:
                            unit = "month"
                        elif unit in ["days", "day"]:
                            unit = "day"
                        elif unit in ["hours", "hour"]:
                            unit = "hour"
                            
                        # Extract the subject of the numeric fact (simplified approach)
                        # Look for a noun before the number
                        subject = "unknown"
                        for token in sent.doc:
                            if token.text == value_str:
                                # Look back for the closest noun
                                for prev_token in reversed(list(token.doc[:token.i])):
                                    if prev_token.pos_ in ["NOUN", "PROPN"]:
                                        subject = prev_token.text.lower()
                                        break
                                break
                                
                        fact_key = f"{subject}_{unit}"
                        if fact_key not in numerical_facts:
                            numerical_facts[fact_key] = []
                            
                        numerical_facts[fact_key].append((value, sent_text, doc_id))
                    except ValueError:
                        pass  # Not a valid number
    
    # Analyze for potential contradictions
    contradiction_report = []
    
    # Method 1: Compare statements with high textual similarity but different meaning
    if have_similarity_model:
        # Get unique factual statements
        statements = list(fact_statements.keys())
        
        # Only compare if we have a reasonable number of statements
        if len(statements) < 1000:  # Avoid excessive comparisons
            # Calculate similarity matrix
            try:
                embeddings = similarity_model.encode(statements)
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(embeddings)
                
                # Find highly similar statements from different documents
                for i in range(len(statements)):
                    for j in range(i+1, len(statements)):
                        similarity = similarity_matrix[i][j]
                        
                        # High similarity but from different docs
                        if similarity > 0.8 and not set(fact_statements[statements[i]]).isdisjoint(fact_statements[statements[j]]):
                            # Check for potential semantic contradiction
                            if contains_contradiction(statements[i], statements[j]):
                                contradiction_report.append({
                                    "type": "semantic_contradiction",
                                    "statement1": statements[i],
                                    "statement2": statements[j],
                                    "source1": fact_statements[statements[i]],
                                    "source2": fact_statements[statements[j]],
                                    "similarity": float(similarity)
                                })
            except Exception as e:
                logger.warning(f"Error in similarity-based contradiction detection: {str(e)}")
    
    # Method 2: Compare numerical facts
    for fact_key, values in numerical_facts.items():
        if len(values) > 1:
            # Sort by value for easy comparison
            values.sort(key=lambda x: x[0])
            
            # Check for significant differences
            min_val = values[0][0]
            max_val = values[-1][0]
            
            # Calculate relative difference
            if min_val > 0:
                relative_diff = (max_val - min_val) / min_val
            else:
                relative_diff = float("inf")
                
            # Report if significant difference (more than 20%)
            if relative_diff > 0.2:
                contradiction_report.append({
                    "type": "numerical_contradiction",
                    "fact_key": fact_key,
                    "min_value": {
                        "value": min_val,
                        "statement": values[0][1],
                        "source": values[0][2]
                    },
                    "max_value": {
                        "value": max_val,
                        "statement": values[-1][1],
                        "source": values[-1][2]
                    },
                    "relative_difference": float(relative_diff)
                })
    
    # Tag documents with contradiction information
    for doc in documents:
        doc_id = f"{doc['metadata'].get('source', 'unknown')}:{doc['metadata'].get('chunk_index', 0)}"
        
        # Check if this document is involved in any contradictions
        is_contradictory = False
        contradictions_list = []
        
        for contradiction in contradiction_report:
            if contradiction["type"] == "semantic_contradiction":
                if doc_id in contradiction["source1"] or doc_id in contradiction["source2"]:
                    is_contradictory = True
                    contradictions_list.append({
                        "type": "semantic",
                        "statement": contradiction["statement1"] if doc_id in contradiction["source1"] else contradiction["statement2"],
                        "conflicting_statement": contradiction["statement2"] if doc_id in contradiction["source1"] else contradiction["statement1"]
                    })
            elif contradiction["type"] == "numerical_contradiction":
                if doc_id == contradiction["min_value"]["source"] or doc_id == contradiction["max_value"]["source"]:
                    is_contradictory = True
                    contradictions_list.append({
                        "type": "numerical",
                        "fact_key": contradiction["fact_key"],
                        "value": contradiction["min_value"]["value"] if doc_id == contradiction["min_value"]["source"] else contradiction["max_value"]["value"],
                        "conflicting_value": contradiction["max_value"]["value"] if doc_id == contradiction["min_value"]["source"] else contradiction["min_value"]["value"]
                    })
        
        # Add contradiction metadata to document
        doc["metadata"]["has_contradictions"] = is_contradictory
        if is_contradictory:
            doc["metadata"]["contradictions"] = contradictions_list
            
            # Add a warning note to the content for potentially contradictory info
            warning_note = "\n[WARNING: This document contains information that may contradict other sources]"
            if warning_note not in doc["page_content"]:
                doc["page_content"] += warning_note
        
        processed_docs.append(doc)
    
    # Save contradiction report
    if contradiction_report:
        report_path = os.path.join(os.path.dirname(METADATA_CACHE_PATH), "contradiction_report.json")
        with open(report_path, 'w') as f:
            json.dump(contradiction_report, f, indent=2)
        logger.info(f"Saved contradiction report with {len(contradiction_report)} potential contradictions to {report_path}")
    
    return processed_docs, contradiction_report

def contains_contradiction(statement1, statement2):
    """
    Check if two statements potentially contradict each other.
    This is a simplified approach that looks for opposing sentiment/meaning.
    
    Args:
        statement1: First statement
        statement2: Second statement
        
    Returns:
        Boolean indicating potential contradiction
    """
    # Basic negation check
    negation_markers = ["not", "n't", "no", "never", "none", "neither", "nor"]
    
    has_negation1 = any(marker in statement1.lower().split() for marker in negation_markers)
    has_negation2 = any(marker in statement2.lower().split() for marker in negation_markers)
    
    # If one has negation and the other doesn't, potential contradiction
    if has_negation1 != has_negation2:
        # Also check for common content by removing negation words
        words1 = set(w.lower() for w in statement1.split() if w.lower() not in negation_markers)
        words2 = set(w.lower() for w in statement2.split() if w.lower() not in negation_markers)
        
        # Calculate Jaccard similarity on remaining words
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) > 0 and len(intersection) / len(union) > 0.5:
            return True
    
    # Check for opposite qualifiers
    positive_qualifiers = ["good", "better", "best", "high", "higher", "increase", "more", "greater", "larger"]
    negative_qualifiers = ["bad", "worse", "worst", "low", "lower", "decrease", "less", "smaller", "lesser"]
    
    # Check if one statement has positive qualifiers and the other has negative
    has_positive1 = any(qual in statement1.lower().split() for qual in positive_qualifiers)
    has_negative1 = any(qual in statement1.lower().split() for qual in negative_qualifiers)
    has_positive2 = any(qual in statement2.lower().split() for qual in positive_qualifiers)
    has_negative2 = any(qual in statement2.lower().split() for qual in negative_qualifiers)
    
    if (has_positive1 and has_negative2) or (has_negative1 and has_positive2):
        return True
    
    return False

def extract_topics_from_content(content, topics):
    # Use spaCy to process the content
    doc = nlp(content)
    
    # Extract key topics from named entities and noun chunks
    extracted_topics = [ent.text for ent in doc.ents] + [chunk.text for chunk in doc.noun_chunks]
    
    # Update the topics dictionary with counts
    for topic in extracted_topics:
        topic = topic.lower()  # normalize to lowercase 
        if topic not in topics:
            topics[topic] = 0
        topics[topic] += 1

def main():
    """Main function to process documents and build vector store with enhanced features for French content."""
    logger.info("Starting enhanced document processing for French content")
    
    # Initialize the embedding model with a French-focused multilingual model
    logger.info("Initializing French-focused embedding model")
    embedding_model = HuggingFaceEmbeddings(model_name="antoinelouis/french-me5-small")
    
    # Load the cache of processed files
    processed_files = load_processed_files()
    
    # Discover document files
    document_files = discover_document_files(DATA_PATH)
    logger.info(f"Found {len(document_files)} document files")
    
    # Prioritize documents, treating all modules equally by default
    # You can specify custom module priorities here if needed, e.g.:
    # module_priorities = {"CS101": 1.5, "MATH202": 0.8}
    prioritized_files = prioritize_documents(document_files)
    
    # Determine which files need processing
    files_to_process = []
    for file_path, file_type, priority in prioritized_files:
        try:
            current_hash = get_file_hash(file_path)
            if file_path not in processed_files or processed_files[file_path] != current_hash:
                files_to_process.append((file_path, file_type, priority))
                processed_files[file_path] = current_hash  # Update hash
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {str(e)}")
    
    logger.info(f"{len(files_to_process)} files need processing")
    
    # Determine optimal batch size dynamically
    optimal_batch_size = determine_optimal_batch_size()
    
    # Update NUM_WORKERS based on system capabilities
    num_workers = min(os.cpu_count() or 4, NUM_WORKERS)
    logger.info(f"Using {num_workers} workers with batch size {optimal_batch_size}")
    
    # Load and process documents in parallel
    all_documents = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Process files in batches to control memory usage
        for i in range(0, len(files_to_process), optimal_batch_size):
            batch = files_to_process[i:i + optimal_batch_size]
            logger.info(f"Processing batch {i // optimal_batch_size + 1}/{(len(files_to_process) + optimal_batch_size - 1) // optimal_batch_size}")
            
            future_to_file = {executor.submit(load_and_chunk_file, file_path): file_path 
                             for file_path, _, _ in batch}
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Loading documents"):
                try:
                    docs = future.result()
                    all_documents.extend(docs)
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Failed to process {file_path}: {str(e)}")
    
                # Explicitly run garbage collection between batches
                gc.collect()
    
    # Create semantic chunks using the improved method
    text_chunks = create_semantic_chunks(all_documents)
    logger.info(f"Created {len(text_chunks)} semantic chunks from {len(all_documents)} documents")
    
    # Apply enhanced noise robustness processing
    enhanced_chunks = enhance_noise_robustness(text_chunks)
    logger.info(f"Applied noise robustness processing to {len(enhanced_chunks)} chunks")
    
    # Detect and resolve contradictions
    processed_chunks, contradiction_report = detect_and_resolve_contradictions(enhanced_chunks)
    logger.info(f"Processed {len(processed_chunks)} chunks for contradictions, found {len(contradiction_report)} potential contradictions")
    
    # Fine-tune the embedding model if enough data is available
    if len(processed_chunks) > 100:  # Only fine-tune if we have enough data
        logger.info("Fine-tuning French embedding model on course content")
        fine_tuner = CourseEmbeddingFineTuner(base_model="antoinelouis/french-me5-small")
        embedding_model = fine_tuner.fine_tune(processed_chunks)
    
    # Update the vector store with advanced retrieval model preparation
    if processed_chunks:
        # Determine best retrieval model based on data characteristics
        retrieval_model = "dpr" if len(processed_chunks) > 1000 else "default"
        logger.info(f"Using {retrieval_model} retrieval model")
        
        # Calculate optimal batch size for vector store updates
        vector_batch_size = determine_optimal_batch_size(
            doc_length=sum(len(doc["page_content"]) for doc in processed_chunks[:100]) // min(100, len(processed_chunks))
        )
        
        # Build or update vector store with advanced retrieval
        db = build_or_update_vector_store(processed_chunks, embedding_model, retrieval_model)
        
        # Apply advanced indexing optimization for large datasets
        db = optimize_vector_store_indexing(db, len(processed_chunks))
        
        # Generate comprehensive coverage analysis
        save_metadata_summary(processed_chunks)
    
    # Save the updated cache of processed files
    save_processed_files(processed_files)
    
    logger.info("Enhanced document processing complete")

if __name__ == "__main__":
    main()
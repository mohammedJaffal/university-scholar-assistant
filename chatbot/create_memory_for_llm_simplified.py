from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "docs"                      # Where your study material lives
FAISS_DIR = "vectorstore/db_faiss"     # Where to write the index
PROCESSED_CACHE = "vectorstore/processed_files.json"  # File‑hash cache
CHUNK_SIZE = 800                       # Approx. tokens per chunk
CHUNK_OVERLAP = 80                     # Token overlap between chunks
MAX_WORKERS = min(4, (os.cpu_count() or 4))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("doc_ingest")

def sha256(path: str) -> str:
    """Return the SHA‑256 hash of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB blocks
            h.update(chunk)
    return h.hexdigest()


def load_cache() -> Dict[str, str]:
    if os.path.exists(PROCESSED_CACHE):
        return json.load(open(PROCESSED_CACHE))
    return {}


def save_cache(cache: Dict[str, str]) -> None:
    os.makedirs(Path(PROCESSED_CACHE).parent, exist_ok=True)
    json.dump(cache, open(PROCESSED_CACHE, "w"))


EXT2LOADER = {
    ".pdf": PyPDFLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


def discover(root: str) -> List[str]:
    """Return every document path under *root* matching the extensions above plus .txt."""
    patterns = [f"**/*{ext}" for ext in EXT2LOADER] + ["**/*.txt"]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(root, pattern), recursive=True))
    return files


SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
)


def load_and_split(path: str) -> List[Dict]:
    """Load *path* with an appropriate loader, then split into chunks with metadata."""
    ext = Path(path).suffix.lower()
    loader_cls = EXT2LOADER.get(ext, UnstructuredFileLoader)
    loader = loader_cls(path)  # type: ignore[arg-type]
    docs = loader.load()

    chunks: List[Dict] = []
    for doc in docs:
        for chunk in SPLITTER.split_text(doc.page_content):
            chunks.append(
                {
                    "page_content": chunk,
                    "metadata": {
                        "source": path,
                        "chunk": len(chunks),
                    },
                }
            )
    return chunks


def build_or_update(chunks: List[Dict], embed) -> FAISS:
    texts = [c["page_content"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    if os.path.exists(FAISS_DIR):
        log.info("Updating existing FAISS index …")
        db = FAISS.load_local(
            FAISS_DIR, embed, allow_dangerous_deserialization=True
        )
        db.add_texts(texts, metas, batch_size=32)
    else:
        log.info("Creating new FAISS index …")
        db = FAISS.from_texts(texts, embed, metadatas=metas)

    db.save_local(FAISS_DIR)
    return db


def main() -> None:
    cache = load_cache()

    files = discover(DATA_DIR)
    todo = [f for f in files if cache.get(f) != sha256(f)]

    if not todo:
        log.info("Nothing to do – all files up‑to‑date.")
        return

    log.info("Embedding model: intfloat/multilingual-e5-small")
    embed = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    chunks: List[Dict] = []
    with ThreadPoolExecutor(MAX_WORKERS) as tpe:
        futures = {tpe.submit(load_and_split, p): p for p in todo}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                res = fut.result()
                chunks.extend(res)
                cache[path] = sha256(path)
                log.info(f"{Path(path).name}: {len(res)} chunks")
            except Exception as e:
                log.error(f"{path}: {e}")

    if chunks:
        log.info(f"Storing {len(chunks)} new chunks …")
        build_or_update(chunks, embed)
        save_cache(cache)
    else:
        log.info("No new chunks generated.")


if __name__ == "__main__":
    main()

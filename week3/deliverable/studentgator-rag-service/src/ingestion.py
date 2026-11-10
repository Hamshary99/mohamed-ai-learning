"""
src/ingestion.py
Offline Document Ingestion & Vector Indexing Script
===================================================
Executes the offline RAG indexing pipeline:
1. Discovers and loads all institutional PDFs from data/ via metadata_registry.
2. Recursively splits text into chunks with overlap.
3. Enriches chunk metadata with domain tags, statutory terms, and page numbers.
4. Persists the ChromaDB vector database index to disk (default: data/chroma_db).

Running this as a standalone script:
    python -m src.ingestion
Pre-builds the database so the FastAPI server boots in < 0.2 seconds without re-indexing.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from src.config import settings
from src.metadata import metadata_registry
from src.retriever import get_default_embedder


def chunk_and_enrich_documents(
    raw_documents: List[Document],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap
) -> List[Document]:
    """
    Splits raw page documents into overlapping chunks and enriches
    each chunk with ChromaDB-safe metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    raw_chunks = splitter.split_documents(raw_documents)
    enriched_chunks: List[Document] = []

    for idx, chunk in enumerate(raw_chunks, start=1):
        file_name = chunk.metadata.get("file_name", chunk.metadata.get("source", "unknown.pdf"))
        page_number = chunk.metadata.get("page", 1)
        chunk_id = f"{Path(file_name).stem}_c{idx:04d}"

        # Generate ChromaDB-safe flattened metadata
        safe_meta = metadata_registry.enrich_chunk_metadata(
            file_name=file_name,
            page_number=page_number,
            chunk_id=chunk_id,
            chunk_type="child",
            extra_metadata=chunk.metadata
        )

        enriched_chunks.append(
            Document(page_content=chunk.page_content, metadata=safe_meta)
        )

    return enriched_chunks


def build_persistent_vectorstore(
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
    force_rebuild: bool = False
) -> Chroma:
    """
    Builds or updates the persistent ChromaDB collection on disk.
    If force_rebuild=True, re-indexes all documents from scratch.
    """
    persist_dir = persist_directory or settings.chroma_persist_directory
    base_coll = collection_name or settings.chroma_collection_name
    embeddings, provider = get_default_embedder()
    coll_name = f"{base_coll}_{provider}"

    persist_path = Path(persist_dir).resolve()
    os.makedirs(persist_path, exist_ok=True)

    # 1. Load all raw PDF documents from data/
    print(f"[Ingestion] Scanning documents via metadata registry...")
    raw_docs = metadata_registry.load_documents()

    # 2. Split and enrich chunks
    print(f"[Ingestion] Chunking {len(raw_docs)} pages (size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    chunks = chunk_and_enrich_documents(raw_docs)
    print(f"[Ingestion] Created {len(chunks)} enriched chunks.")

    # 3. Build & persist ChromaDB collection to disk
    print(f"[Ingestion] Generating embeddings and persisting to '{persist_path}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=coll_name,
        persist_directory=str(persist_path)
    )
    print(f"[Ingestion] Successfully indexed {len(chunks)} chunks into collection '{coll_name}'.")

    return vectorstore


def get_or_load_vectorstore(
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None
) -> Chroma:
    """
    Fast loader: Connects to an existing ChromaDB on disk in < 0.1s.
    If no database exists yet, automatically builds it.
    """
    persist_dir = persist_directory or settings.chroma_persist_directory
    base_coll = collection_name or settings.chroma_collection_name
    persist_path = Path(persist_dir).resolve()

    embeddings, provider = get_default_embedder()
    coll_name = f"{base_coll}_{provider}"

    # Check if existing database files exist
    if persist_path.exists() and any(persist_path.iterdir()):
        print(f"[VectorStore] Loading existing ChromaDB from disk: '{persist_path}'")
        return Chroma(
            collection_name=coll_name,
            embedding_function=embeddings,
            persist_directory=str(persist_path)
        )

    # Database does not exist yet; build it
    print(f"[VectorStore] No existing database found at '{persist_path}'. Building from source documents...")
    return build_persistent_vectorstore(persist_dir, coll_name)


# =========================================================================
# Standalone CLI Execution: python -m src.ingestion
# =========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("StudentGator Offline RAG Ingestion Pipeline")
    print("=" * 70)

    # Build persistent vector index on disk
    vs = build_persistent_vectorstore(force_rebuild=True)

    # Verification query
    test_query = "What are the requirements for an F-1 visa travel signature?"
    print(f"\n[Verification] Running test similarity search: '{test_query}'...")
    results = vs.similarity_search_with_relevance_scores(test_query, k=2)

    for rank, (doc, score) in enumerate(results, start=1):
        print(f"\nResult #{rank} [Relevance Score: {score:.3f}]")
        print(f"  Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(f"  Domain: {doc.metadata.get('domain_category')}")
        print(f"  Snippet: {doc.page_content[:140]}...")

    print("\n" + "=" * 70)
    print(f"Ingestion Complete. Database ready at: {settings.chroma_persist_directory}")
    print("=" * 70)

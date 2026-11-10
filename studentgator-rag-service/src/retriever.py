"""
src/retriever.py
Hybrid Retrieval & Cross-Encoder Re-Ranking Service
===================================================
Pipeline Architecture:
1. Recursive Character Chunking (400 chars, 60 overlap).
2. Dense Semantic Vector Store (ChromaDB + OpenAI Embeddings).
3. Sparse Lexical Search (BM25 keyword retriever).
4. Weighted Ensemble Retriever (Reciprocal Rank Fusion - 60% Dense / 40% Sparse).
5. Cross-Encoder Re-Ranker (SentenceTransformers joint token attention).
"""

import os
from pathlib import Path
from typing import List, Optional, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from src.config import settings

# In LangChain 0.2+, EnsembleRetriever lives in langchain_classic.retrievers
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    # pyrefly: ignore[missing-import]
    from langchain.retrievers import EnsembleRetriever


def get_default_embedder():
    """
    Factory creating the embedding engine with automatic local Hugging Face fallback.
    Probes cloud providers (Gemini / OpenAI) first; if API keys are corrupt, revoked,
    or unreachable, automatically falls back to local HuggingFace embeddings.
    """
    # If explicitly configured for local embeddings (recommended to avoid API rate limits)
    if "minilm" in settings.embedding_model.lower():
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), "hf"

    gemini_key = settings.gemini_api_key or settings.google_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            emb_model = settings.embedding_model if "gemini" in settings.embedding_model else "models/gemini-embedding-001"
            embedder = GoogleGenerativeAIEmbeddings(model=emb_model, google_api_key=gemini_key)
            embedder.embed_query("probe")
            return embedder, "gemini"
        except Exception as exc:
            print(f"[Embeddings Fallback] Gemini API validation failed ({exc}). Falling back to local Hugging Face (all-MiniLM-L6-v2)...")

    if settings.openai_api_key or os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings
            embedder = OpenAIEmbeddings(model=settings.embedding_model)
            embedder.embed_query("probe")
            return embedder, "openai"
        except Exception as exc:
            print(f"[Embeddings Fallback] OpenAI API validation failed ({exc}). Falling back to local Hugging Face (all-MiniLM-L6-v2)...")

    from langchain_huggingface import HuggingFaceEmbeddings
    print("[Embeddings] Using offline local Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2).")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), "hf"


class AdvancedRetrieverService:
    def __init__(
        self,
        raw_documents: List[Document],
        embedder: Optional[Any] = None,
        collection_name: str = "studentgator_knowledge_base"
    ):
        # 1. Recursive Character Chunking
        # Splits on paragraphs (\n\n), lines (\n), sentences (. ), and words
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.chunks = splitter.split_documents(raw_documents)

        # 2. Dense Semantic Vector Store (ChromaDB)
        if embedder is not None:
            self.embeddings = embedder
            provider = "custom"
        else:
            self.embeddings, provider = get_default_embedder()

        target_coll = f"{collection_name}_{provider}"
        persist_dir = Path(settings.chroma_persist_directory).resolve()
        os.makedirs(persist_dir, exist_ok=True)

        self.vectorstore = Chroma(
            collection_name=target_coll,
            embedding_function=self.embeddings,
            persist_directory=str(persist_dir)
        )

        # If the target collection is empty in persistent storage, index the chunks
        try:
            doc_count = self.vectorstore._collection.count()
        except Exception:
            doc_count = 0

        if doc_count == 0 and self.chunks:
            print(f"[VectorStore] Indexing {len(self.chunks)} chunks into Chroma collection '{target_coll}'...")
            self.vectorstore.add_documents(self.chunks)
        else:
            print(f"[VectorStore] Connected to existing Chroma collection '{target_coll}' ({doc_count} chunks).")

        # Dense search: retrieves the top 6 semantically closest chunks
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})

        # 3. Sparse Lexical Search (BM25)
        # If no raw_documents were passed, populate BM25 directly from the persistent vector store
        if not self.chunks:
            stored_data = self.vectorstore.get()
            stored_docs = stored_data.get("documents", [])
            stored_metas = stored_data.get("metadatas", [])
            self.chunks = [
                Document(page_content=d, metadata=m or {})
                for d, m in zip(stored_docs, stored_metas)
            ]

        if self.chunks:
            self.sparse_retriever = BM25Retriever.from_documents(self.chunks)
            self.sparse_retriever.k = settings.retrieval_k
        else:
            self.sparse_retriever = None

        # 4. Weighted Ensemble Retriever (Reciprocal Rank Fusion)
        if self.sparse_retriever is not None:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.dense_retriever, self.sparse_retriever],
                weights=[0.6, 0.4]
            )
        else:
            self.ensemble_retriever = self.dense_retriever

        # 5. Cross-Encoder Re-Ranker
        # Lightweight re-ranker for joint query-document cross-attention scoring
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def retrieve_and_rerank(self, query: str, top_k: int = 3, min_score: float = -2.0) -> List[Document]:
        """
        Two-stage retrieval funnel with cross-encoder relevance gating:
        - Stage 1: Coarse candidate recall via Ensemble RRF (Dense + BM25).
        - Stage 2: Fine-grained token cross-attention scoring via Cross-Encoder.
        - Stage 3: Strict threshold cutoff (prunes out-of-scope / distractor chunks with scores < min_score).
        """
        # Stage 1: Ensemble RRF candidate recall
        candidates = self.ensemble_retriever.invoke(query)
        if not candidates:
            return []

        # Stage 2: Joint token cross-attention scoring
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(candidates, scores):
            doc.metadata["relevance_score"] = float(score)

        # Sort descending by re-rank score
        ranked = sorted(candidates, key=lambda d: d.metadata["relevance_score"], reverse=True)

        # Stage 3: Prune irrelevant cross-border chunks below minimum relevance cutoff
        filtered = [doc for doc in ranked if doc.metadata["relevance_score"] >= min_score]
        return filtered[:top_k]

    # Backward compatibility alias
    retreiver_and_reranker = retrieve_and_rerank
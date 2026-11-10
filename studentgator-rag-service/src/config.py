"""
src/config.py
Pydantic v2 Settings & Environment Configuration
================================================
Centralizes all application configurations, API keys, database paths,
and model hyperparameters loaded from .env or system environment variables.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root path (directory containing .env and data/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Strongly-typed application settings.
    Automatically reads from .env file or system environment variables.
    """
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # 1. Model Provider Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key for LLMs & Embeddings")
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API Key")
    google_api_key: Optional[str] = Field(default=None, description="Google Gemini API Key alias")
    hf_token: Optional[str] = Field(default=None, description="Hugging Face API Token")

    # 2. LangSmith Observability
    langsmith_tracing: bool = Field(default=False, description="Enable LangSmith distributed tracing")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API Key")
    langsmith_project: str = Field(default="studentgator-week3-capstone", description="LangSmith tracing project name")

    # 3. Vector Database & Storage
    chroma_persist_directory: str = Field(
        default=str(PROJECT_ROOT / "data" / "chroma_db"),
        description="Path on disk where ChromaDB persists vector embeddings"
    )
    chroma_collection_name: str = Field(
        default="studentgator_knowledge_base",
        description="ChromaDB collection / table name"
    )

    # 4. Model Hyperparameters
    default_model: str = Field(default="gemini-3.5-flash", description="Primary chat model for answers & routing")
    default_temperature: float = Field(default=0.0, description="Temperature for deterministic factual RAG")
    embedding_model: str = Field(default="models/gemini-embedding-001", description="Dense embedding model")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="HuggingFace Cross-Encoder model")

    # 5. Chunking & Retrieval Parameters
    chunk_size: int = Field(default=400, description="Recursive character text splitter chunk size")
    chunk_overlap: int = Field(default=60, description="Chunk overlap to preserve boundary context")
    retrieval_k: int = Field(default=6, description="Candidate chunks pulled per retriever stream (Chroma and BM25)")
    rerank_top_k: int = Field(default=3, description="Final high-precision chunks passed to the LLM prompt")

    # 6. API Server Settings
    api_host: str = Field(default="0.0.0.0", description="FastAPI host interface")
    api_port: int = Field(default=8000, description="FastAPI port")


# Global singleton instance for easy import across modules
# Usage: from src.config import settings
settings = Settings()

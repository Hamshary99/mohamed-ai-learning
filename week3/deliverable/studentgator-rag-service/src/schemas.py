"""
src/schemas.py
Pydantic v2 Request and Response Data Contracts
===============================================
Defines strict API boundaries between frontend clients and the agentic backend.
Pydantic validates and sanitizes payloads at the gateway, returning HTTP 422
Unprocessable Entity before any model tokens or database queries are executed.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


# =========================================================================
# 1. Incoming Request Contracts
# =========================================================================

class ChatRequest(BaseModel):
    """
    Inbound payload submitted by the student chat interface.
    Acts as a gatekeeper contract to protect the LangGraph pipeline from malformed inputs.
    """
    # - extra="forbid": Rejects unapproved JSON keys (blocks payload injection / catches typos).
    # - str_strip_whitespace=True: Trims whitespace so "   " becomes "" and fails min_length=1.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Required user prompt:
    # - ... (Ellipsis): Field cannot be omitted.
    # - min_length=1: Prevents empty prompts from triggering wasted LLM calls.
    # - max_length=2000: Caps input size to protect against prompt-bomb token exhaustion.
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query or conversational message submitted to the agent."
    )

    # Session identifier for LangGraph persistence:
    # Passed to LangGraph's checkpointer to load and append to the student's conversation history.
    thread_id: str = Field(
        default="default_session",
        description="Unique session identifier for checkpoint state persistence."
    )

    # Optional metadata pre-filter:
    # Allows the UI to restrict retrieval to a specific country or domain (e.g. 'eu_germany_admissions').
    category_filter: Optional[str] = Field(
        default=None,
        description="Optional institutional domain category to pre-filter ChromaDB search."
    )


# Backward compatibility alias in case Chatrequest was imported elsewhere
Chatrequest = ChatRequest


# =========================================================================
# 2. Outgoing Response Contracts (Commented for now)
# =========================================================================

# class CitationMetadata(BaseModel):
#     """
#     Structured citation reference returned with answers so students can verify sources.
#     """
#     source: str = Field(..., description="PDF source file name (e.g. 'eu_germany_study_visa_regulations.pdf')")
#     page: Optional[int] = Field(default=None, description="Page number where the cited text was found")
#     domain_category: Optional[str] = Field(default=None, description="Categorical domain tag of the source")
#     relevance_score: Optional[float] = Field(default=None, description="Cross-encoder relevance score")
# 
# 
# class ChatResponse(BaseModel):
#     """
#     Standard non-streaming JSON response contract returned to HTTP clients.
#     """
#     thread_id: str = Field(..., description="Active session thread ID")
#     answer: str = Field(..., description="Final grounded answer synthesized by the agent")
#     citations: List[CitationMetadata] = Field(default_factory=list, description="Verified source document citations")


# =========================================================================
# 3. Infrastructure & Streaming Event Contracts
# =========================================================================

class HealthResponse(BaseModel):
    """
    Health check contract returned by GET /health.
    Used by Docker, Kubernetes, or Cloud Run probes to confirm the service and its
    backing indexes (Vector DB, LangSmith) are initialized before sending live traffic.
    """
    model_config = ConfigDict(populate_by_name=True)

    status: str = Field(default="healthy", description="Overall service status ('healthy' or 'degraded')")
    vectorstore_indexed: bool = Field(default=False, alias="vectorstore_index", description="True if ChromaDB index is loaded")
    langsmith_tracing: bool = Field(..., description="True if LangSmith observability tracing is active")


# Structured SSE Event Schemas (Commented for now)
# class TokenEvent(BaseModel):
#     """Emitted token-by-token during live LLM text generation."""
#     type: str = "token"
#     content: str
# 
# 
# class CitationEvent(BaseModel):
#     """Emitted once retrieval finishes so the UI displays source badges immediately."""
#     type: str = "citation"
#     source: str
#     page: Optional[int] = None
#     domain_category: Optional[str] = None
# 
# 
# class DoneEvent(BaseModel):
#     """Emitted at the end of the stream to signal completion."""
#     type: str = "done"
#     total_tokens: int

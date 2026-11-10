"""
src/main.py
FastAPI Asynchronous Microservice with Server-Sent Events Token Streaming
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from src.schemas import ChatRequest, HealthResponse
from src.retriever import AdvancedRetrieverService
from src.graph.workflow import build_studentgator_graph
from src.metadata import metadata_registry

# Global service singletons
retriever_service = None
rag_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever_service, rag_graph
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "studentgator-week3-capstone")

    print("[Startup] Initializing StudentGator RAG Service...")
    # Reference institutional documentation offloaded to metadata registry
    # Automatically discovers and loads all PDFs from the data/ directory.
    raw_institutional_docs = metadata_registry.load_documents()

    # Initialize indexing and graph execution pipeline
    retriever_service = AdvancedRetrieverService(raw_institutional_docs)
    rag_graph = build_studentgator_graph(retriever_service)
    yield
    retriever_service = None
    rag_graph = None


# FastAPI application instance defined at module level
app = FastAPI(
    title="StudentGator Production Agentic RAG Service",
    description="Asynchronous LangGraph RAG Service with Hybrid Search, Cross-Encoder Re-Ranking, and SSE Streaming.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_graph():
    if rag_graph is None:
        raise HTTPException(status_code=503, detail="RAG Graph not initialized. Please try again later.")
    return rag_graph


async def event_stream_generator(query: str, thread_id: str, request: Request) -> AsyncGenerator[str, None]:
    """
    Streams token events over Server-Sent Events (SSE) using LangGraph's astream_events API.
    """
    graph = get_graph()
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "current_query": query,
        "intent": None,
        "retrieved_docs": [],
        "final_response": ""
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # 1. Initial SSE handshake event
        yield f"event: start\ndata: {json.dumps({'status': 'processing', 'query': query})}\n\n"

        # 2. Stream graph events as they execute across nodes
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            # Check for client disconnect to avoid wasting LLM resources
            if await request.is_disconnected():
                break

            # In LangGraph astream_events v2, event type is in event["event"]
            event_type = event.get("event")

            # Surface routing intent to client (e.g. 'greeting' vs 'institutional_rag')
            if event_type in ("on_node_end", "on_chain_end") and event.get("name") == "classify":
                intent = (event.get("data") or {}).get("output", {}).get("intent")
                if intent:
                    yield f"event: route\ndata: {json.dumps({'intent': intent})}\n\n"

            # Stream retrieved sources event (for terminal logging in clients)
            elif event_type in ("on_node_end", "on_chain_end") and event.get("name") == "rag_retrieval_node":
                output_data = (event.get("data") or {}).get("output", {})
                docs = output_data.get("retrieved_docs", [])
                source_list = [
                    {
                        "source": d.metadata.get("source", "Unknown"),
                        "page": d.metadata.get("page", "N/A"),
                        "score": round(d.metadata.get("relevance_score", 0.0), 3)
                    }
                    for d in docs
                ]
                if source_list:
                    yield f"event: sources\ndata: {json.dumps({'sources': source_list})}\n\n"

            # Stream direct conversational responses (bypasses LLM, e.g. greeting or out-of-scope)
            elif event_type in ("on_node_end", "on_chain_end") and event.get("name") == "direct_node":
                output_data = (event.get("data") or {}).get("output", {})
                direct_reply_text = output_data.get("final_response", "")
                if direct_reply_text:
                    payload = json.dumps({"token": direct_reply_text})
                    yield f"event: token\ndata: {payload}\n\n"

            # Stream LLM tokens live as they are generated
            elif event_type == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    token_str = ""
                    if isinstance(chunk.content, str):
                        token_str = chunk.content
                    elif isinstance(chunk.content, list):
                        for part in chunk.content:
                            if isinstance(part, str):
                                token_str += part
                            elif isinstance(part, dict) and "text" in part:
                                token_str += part.get("text", "")
                    if token_str:
                        payload = json.dumps({"token": token_str})
                        yield f"event: token\ndata: {payload}\n\n"

        # 3. Stream completion sentinel (sent after the async for loop completes)
        yield "event: done\ndata: [DONE]\n\n"

    except asyncio.CancelledError:
        # Client closed connection or task was cancelled
        return
    except Exception as exc:
        # Catch errors during generation and stream a structured error event
        error_payload = json.dumps({"error": str(exc)})
        yield f"event: error\ndata: {error_payload}\n\n"
        yield "event: done\ndata: [DONE]\n\n"


# Static files directory for interactive web chat UI
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# =========================================================================
# API Endpoints
# =========================================================================

@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
async def serve_chat_ui():
    """Serves the interactive web chat interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>StudentGator RAG Service running. Static UI not found.</h1>")


@app.post("/api/v1/chat")
async def chat_streaming_endpoint(request: Request, payload: ChatRequest):
    """
    Exposes streaming chat completion via Server-Sent Events (SSE).
    """
    return StreamingResponse(
        event_stream_generator(payload.message, payload.thread_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        vectorstore_indexed=retriever_service is not None,
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    )


# =========================================================================
# Local Direct Runner: python src/main.py
# =========================================================================

if __name__ == "__main__":
    import uvicorn
    print("\nStarting StudentGator RAG Server on http://localhost:8000")
    print("Interactive Swagger UI API Docs available at: http://localhost:8000/docs\n")
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
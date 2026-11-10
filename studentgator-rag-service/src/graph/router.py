"""
src/graph/router.py
Pure Decision Router Functions (The Navigators)
================================================
In LangGraph, routers are decision-only functions.
Key Rule: Routers NEVER mutate or write to state!
They only inspect state and return a string naming the next node to trigger.
"""

from typing import Literal
from src.graph.state import AgentState


def route_by_intent(state: AgentState) -> Literal["direct_node", "rag_retrieval_node"]:
    """
    Inspects state['intent'] and steers the graph execution path.

    How LangGraph uses this:
    - If it returns "direct_node": LangGraph routes to direct_reply_node (fast, zero vector DB calls).
    - If it returns "rag_retrieval_node": LangGraph routes to retrieval_node (runs hybrid search + re-ranking).

    Mental Model:
    This is like an airport flight navigator. It doesn't fly the plane or load luggage (that's what
    nodes do); it only looks at the weather tag and points the plane toward Runway A or Runway B.
    """
    intent = state.get("intent", "institutional_rag")

    # Greetings ("hi") and off-topic queries ("pasta recipe") skip the vector database entirely
    if intent in ["greeting", "out_of_scope"]:
        return "direct_node"

    # Serious admission or visa questions proceed to the full RAG pipeline
    return "rag_retrieval_node"

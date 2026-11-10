"""
src/graph/workflow.py
LangGraph Orchestration & Graph Assembly (The Blueprint)
========================================================
This file is the architectural blueprint of the entire AI agent.
Notice how clean and readable it is because nodes and routing logic are decoupled:
- state.py:  Defines the data contract (The Clipboard).
- router.py: Defines pure routing decisions (The Navigator).
- nodes.py:  Defines worker tasks (The Assembly Stations).
- workflow.py: Connects the stations and compiles the pipeline.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import AgentState
from src.graph.nodes import StudentGatorNodes
from src.graph.router import route_by_intent
from src.retriever import AdvancedRetrieverService


def build_studentgator_graph(retriever_service: AdvancedRetrieverService, llm=None):
    """
    Wires nodes, conditional edges, and persistence into a compiled LangGraph workflow.

    Visual Data Flow:
    
    [ START ] ──> [ Node: 'classify' ]
                         │
                         ▼ (Conditional Edge: route_by_intent)
           ┌─────────────┴─────────────┐
           ▼                           ▼
    [ 'direct_node' ]       [ 'rag_retrieval_node' ]
    (Greetings / Out-of-Scope)  (Hybrid Search + Re-rank)
           │                           │
           │                           ▼
           │                 [ 'generate_node' ]
           │                 (Context Prompt Synthesis)
           │                           │
           └─────────────┬─────────────┘
                         ▼
                      [ END ]
    """
    # 1. Initialize our worker nodes with their required dependencies
    nodes = StudentGatorNodes(retriever_service=retriever_service, llm=llm)

    # 2. Initialize the graph using our centralized AgentState schema
    workflow = StateGraph(AgentState)

    # 3. Register the Worker Nodes (add_node gives each function a string name)
    workflow.add_node("classify", nodes.classify_intent)
    workflow.add_node("direct_node", nodes.direct_reply)
    workflow.add_node("rag_retrieval_node", nodes.retrieval)
    workflow.add_node("generate_node", nodes.generate_rag)

    # 4. Wire the Entry Edge: every conversation starts at the 'classify' node
    workflow.add_edge(START, "classify")

    # 5. Wire the Conditional Router Edge:
    # After 'classify' finishes, LangGraph calls route_by_intent(state)
    # and routes to the matching node name in the mapping dictionary.
    workflow.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "direct_node": "direct_node",
            "rag_retrieval_node": "rag_retrieval_node"
        }
    )

    # 6. Wire Progression & Terminal Edges to END
    # - direct_node finishes its greeting/refusal and immediately ends
    workflow.add_edge("direct_node", END)

    # - rag_retrieval_node passes its retrieved chunks directly to generate_node
    workflow.add_edge("rag_retrieval_node", "generate_node")

    # - generate_node produces the final answer and ends the turn
    workflow.add_edge("generate_node", END)

    # 7. Compile with MemorySaver Checkpointer
    # MemorySaver persists state by thread_id across turns in local memory.
    # In production, this can be swapped with PostgresSaver for persistent database storage.
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

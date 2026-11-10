"""
src/graph/nodes.py
State Mutators & Task Workers (The Assembly Line Stations)
==========================================================
In LangGraph, Nodes are the workers that do actual computation:
1. They accept the current `AgentState` as an argument.
2. They perform work (e.g. check keywords, call an LLM, query ChromaDB).
3. They return a dictionary with partial updates (e.g. `{"retrieved_docs": [...]}`).
LangGraph automatically merges those updates back into the centralized state.
"""

import os
from typing import Dict, Any, Optional, List, Iterator
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, AIMessageChunk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk

from src.graph.state import AgentState
from src.retriever import AdvancedRetrieverService
from src.config import settings


class OfflineFallbackChatModel(BaseChatModel):
    """
    100% offline fallback chat model activated whenever cloud LLM API keys
    are corrupt, missing, quota-exhausted, or unreachable.
    Synthesizes a structured answer grounded directly in the retrieved institutional context.
    """

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        context = ""
        user_query = ""
        for m in messages:
            if isinstance(m, SystemMessage) and "INSTITUTIONAL CONTEXT:" in m.content:
                context = m.content.split("INSTITUTIONAL CONTEXT:")[1].strip()
            elif isinstance(m, HumanMessage):
                user_query = m.content

        reply_parts = [
            "[Notice: Running in Local Offline Fallback Mode - Cloud LLM Key Unavailable]",
            "",
            "Based on the official institutional documentation retrieved from your local knowledge base:",
            ""
        ]
        if context:
            # Clean context for chat bubble: omit metadata headers
            clean_lines = []
            for line in context.splitlines():
                if line.strip().startswith("[Source:") or line.strip().startswith("(Policy Reference:"):
                    continue
                clean_lines.append(line)
            clean_text = "\n".join(clean_lines).strip()
            reply_parts.append(clean_text)
        else:
            reply_parts.append("Official policy details for this query were not found in the local documentation.")

        full_reply = "\n".join(reply_parts)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=full_reply))])

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> Iterator[ChatGenerationChunk]:
        res = self._generate(messages, stop=stop, **kwargs)
        text = res.generations[0].message.content
        words = text.split(" ")
        for i, word in enumerate(words):
            chunk_text = word if i == len(words) - 1 else word + " "
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))

    @property
    def _llm_type(self) -> str:
        return "offline-huggingface-fallback"


def _format_message_text(content: Any) -> str:
    """Helper to safely extract pure text from string or structured content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


class StudentGatorNodes:
    """
    Groups all worker nodes in one place and provides shared access to
    dependencies (the LLM and the AdvancedRetrieverService).
    """

    def __init__(
        self,
        retriever_service: AdvancedRetrieverService,
        llm: Optional[Any] = None
    ):
        self.retriever_service = retriever_service
        self.llm = llm or self._build_default_llm()

    def _build_default_llm(self):
        gemini_key = settings.gemini_api_key or settings.google_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                model_name = settings.default_model if "gemini" in settings.default_model else "gemini-3.5-flash"
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=gemini_key,
                    temperature=settings.default_temperature,
                    max_retries=1
                )
                # Quick probe to test key validity & quota availability
                llm.invoke("ping")
                print(f"[LLM] Verified Gemini model '{model_name}'.")
                return llm
            except Exception as exc:
                print(f"[LLM Fallback] Gemini API validation failed ({exc}). Setting up fallback...")

        if settings.openai_api_key or os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=settings.default_model,
                    temperature=settings.default_temperature,
                    max_retries=1
                )
                llm.invoke("ping")
                print(f"[LLM] Verified OpenAI model '{settings.default_model}'.")
                return llm
            except Exception as exc:
                print(f"[LLM Fallback] OpenAI API validation failed ({exc}). Setting up fallback...")

        return self._build_huggingface_fallback_llm()

    def _build_huggingface_fallback_llm(self):
        """Builds a Hugging Face Inference model if HF_TOKEN is present, else local offline model."""
        hf_token = getattr(settings, "hf_token", None) or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if hf_token:
            try:
                from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
                print("[LLM Fallback] Connecting to Hugging Face Inference API...")
                endpoint = HuggingFaceEndpoint(
                    repo_id="meta-llama/Llama-3.2-3B-Instruct",
                    huggingfacehub_api_token=hf_token,
                    temperature=0.01
                )
                return ChatHuggingFace(llm=endpoint)
            except Exception as exc:
                print(f"[LLM Fallback] Hugging Face Endpoint failed ({exc}). Using offline local fallback...")

        print("[LLM Fallback] Engaging offline local structured fallback model.")
        return OfflineFallbackChatModel()

    # -------------------------------------------------------------------------
    # Node 1: Intent Classifier (The Scope Gatekeeper)
    # -------------------------------------------------------------------------
    def classify_intent(self, state: AgentState) -> Dict[str, Any]:
        """
        Station 1 on the assembly line:
        Inspects the raw user query to tag whether this is:
        - "greeting": User said "hello", "thanks", "good morning"
        - "out_of_scope": User asked an off-topic question (recipes, coding, sports)
        - "institutional_rag": User asked a genuine university / visa question

        Why this matters:
        Tagging the intent here allows the router to skip ChromaDB searches for simple greetings,
        saving ~1.5 seconds of latency and GPU compute per request.
        """
        query = state.get("current_query", "").strip().lower()
        cleaned_query = "".join(c for c in query if c.isalnum() or c.isspace()).strip()

        # Check for conversational greetings
        greetings = [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "thanks", "thank you", "sup"
        ]
        if any(cleaned_query == g or cleaned_query.startswith(g + " ") for g in greetings):
            return {"intent": "greeting"}

        # Check for obvious out-of-scope topics
        out_of_scope_keywords = [
            "recipe", "cook", "pasta", "pizza", "burger", "cake", "weather",
            "football", "soccer", "basketball", "movie", "song", "lyrics",
            "joke", "python code", "javascript", "crypto", "bitcoin", "horoscope"
        ]
        if any(k in query for k in out_of_scope_keywords):
            return {"intent": "out_of_scope"}

        # Default: route to the RAG knowledge base
        return {"intent": "institutional_rag"}

    # -------------------------------------------------------------------------
    # Node 2: Direct Conversational Reply (Bypasses Vector Database)
    # -------------------------------------------------------------------------
    def direct_reply(self, state: AgentState) -> Dict[str, Any]:
        """
        Station 2A on the assembly line:
        Executes ONLY when intent is 'greeting' or 'out_of_scope'.
        Returns a fast, friendly response in 0 milliseconds without searching PDFs.

        How state updates work here:
        - 'final_response': The string returned to the user.
        - 'messages': LangGraph's add_messages reducer appends this AIMessage to the chat history.
        """
        intent = state.get("intent", "greeting")

        if intent == "out_of_scope":
            reply = (
                "I am StudentGator's official AI Advisory agent. I specialize in international student "
                "regulations and cannot assist with general off-topic requests.\n\n"
                "I can help you with:\n"
                "• Germany: Section 16b work limits (140 full / 280 half days), blocked accounts, Bürgeramt\n"
                "• United States: F-1 travel rules, OPT / STEM OPT extensions, SEVP portal reporting\n"
                "• France: VLS-TS visa validation on ANEF, 964-hour work limits, CAF housing assistance\n"
                "• Spain: Universidad Carlos III de Madrid (UC3M) degree admissions, TIE/NIE registration\n"
                "• StudentGator: Platform application procedures, document requirements, rejection guidance"
            )
        else:
            reply = (
                "Hello! I am the StudentGator AI Advisor. I can answer questions regarding "
                "F-1 visa travel rules, German study permits, French VLS-TS validation, "
                "health insurance waivers, and university application deadlines. How can I assist you today?"
            )

        return {
            "final_response": reply,
            "messages": [AIMessage(content=reply)]
        }

    # -------------------------------------------------------------------------
    # Node 3: Retrieval & Cross-Encoder Re-Ranking Worker
    # -------------------------------------------------------------------------
    def retrieval(self, state: AgentState) -> Dict[str, Any]:
        """
        Station 2B on the assembly line:
        Executes our Day 2 pipeline:
        1. Hybrid Search (60% Dense ChromaDB + 40% Sparse BM25 via RRF).
        2. Cross-Encoder re-ranking to score candidates by joint token-level attention.
        3. Saves the top 3 verified chunks into state['retrieved_docs'].

        Notice: This node does NOT generate any text. It only finds evidence and clips it
        to the shared state for the next node (generator) to read.
        """
        query = state.get("current_query", "")
        top_chunks = self.retriever_service.retrieve_and_rerank(query, top_k=3)

        # Print top-k sources and scores to the terminal
        if top_chunks:
            print(f"\n[Terminal Log - Retrieved Sources & Scores for Query: \"{query}\"]")
            for idx, doc in enumerate(top_chunks, 1):
                src = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                score = doc.metadata.get("relevance_score")
                score_str = f" | Cross-Encoder Score: {score:+.3f}" if score is not None else ""
                print(f"  {idx}. {src} (Page {page}){score_str}")
            print()
        else:
            print(f"\n[Terminal Log - No sources passed threshold for Query: \"{query}\"]\n")

        # Writes top chunks to 'retrieved_docs' matching AgentState schema
        return {"retrieved_docs": top_chunks}

    # -------------------------------------------------------------------------
    # Node 4: Generation Node (Prompt Synthesis)
    # -------------------------------------------------------------------------
    def generate_rag(self, state: AgentState) -> Dict[str, Any]:
        """
        Station 3 on the assembly line:
        1. Reads state['retrieved_docs'] and formats them into a clean string context with source citations.
        2. Builds a strict system prompt instructing the LLM to answer ONLY from this context.
        3. Calls GPT-4o-mini to synthesize the final grounded response.
        4. Returns the answer string and an AIMessage to append to history.
        """
        query = state.get("current_query", "")
        retrieved_docs = state.get("retrieved_docs", [])
        messages_history = state.get("messages", [])
        if not messages_history:
            messages_history = [HumanMessage(content=query)]
        # Preserve multi-turn dialogue context (last 10 messages)
        recent_history = messages_history[-10:]

        # Format retrieved chunks with their source names, pages, and cross-encoder scores
        context_str = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', 'N/A')} | Score: {doc.metadata.get('relevance_score', 0.0):.3f}]\n{doc.page_content}"
            for doc in retrieved_docs
        ])

        # Strict grounding prompt prevents hallucinations while honoring multi-turn conversation
        prompt = [
            SystemMessage(content=(
                "You are the official StudentGator International Student Advisory AI.\n"
                "Your institutional repository covers: United States (F-1 / OPT regulations), Germany (Section 16b / study visas), "
                "France (VLS-TS validation & student rights), Spain (UC3M admissions & TIE), and StudentGator platform guidelines.\n\n"
                "Operating Guidelines:\n"
                "1. Answer the student's question accurately and authoritatively using the provided institutional context below and recent conversation history.\n"
                "2. Synthesize a complete advisory response: directly answer the inquiry first, then naturally incorporate the relevant conditions, statutory limits, exceptions, and procedural steps found across the retrieved context to provide comprehensive value.\n"
                "3. Do NOT include inline citations or bracketed source tags (such as [Source: ...] or Page numbers) in your chat response. The system logs sources and cross-encoder scores to the terminal separately.\n"
                "4. If a student asks about a country, jurisdiction, or institution NOT in your context (e.g., Canada, UK, Iran, Australia):\n"
                "   - State clearly that official policy details for that country or institution are outside your documentation scope.\n"
                "   - List the regions you do cover (US, Germany, France, Spain, StudentGator).\n"
                "   - Recommend that they consult that country's official immigration department or contact StudentGator support.\n"
                "5. Never fabricate or extrapolate immigration rules beyond the text.\n\n"
                f"INSTITUTIONAL CONTEXT:\n{context_str}"
            )),
            *recent_history
        ]

        # Call the LLM with automatic fallback if primary model fails at runtime
        try:
            response = self.llm.invoke(prompt)
            answer_text = _format_message_text(response.content)
        except Exception as exc:
            print(f"[LLM Error] Primary LLM failed ({exc}). Engaging fallback model...")
            self.llm = self._build_huggingface_fallback_llm()
            fallback_res = self.llm.invoke(prompt)
            answer_text = _format_message_text(fallback_res.content)

        return {
            "final_response": answer_text,
            "messages": [AIMessage(content=answer_text)]  # 'messages' (plural) triggers the add_messages reducer
        }

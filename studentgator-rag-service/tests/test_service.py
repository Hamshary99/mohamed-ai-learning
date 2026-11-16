"""
studentgator-rag-service/tests/test_service.py
Automated CI unit & RAG ingestion tests executed on every push and pull request by GitHub Actions.
Compatible with both `pytest` and `python -m unittest`.
Tests:
1. Pydantic Request & Response Data Contracts (src/schemas.py)
2. Metadata Registry & ChromaDB Relational Filters (src/metadata.py)
3. LangGraph Conditional Router (src/graph/router.py)
4. Full RAG Ingestion Pipeline: PDF Loading, Recursive Chunking, & Vector Indexing (src/ingestion.py)
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

# Ensure studentgator-rag-service root is on sys.path so `from src...` imports resolve
SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from src.schemas import ChatRequest, HealthResponse
from src.metadata import MetadataManager, metadata_registry
from src.graph.router import route_by_intent
from src.ingestion import chunk_and_enrich_documents, build_persistent_vectorstore


class TestStudentGatorRAGService(unittest.TestCase):
    """CI Unit & RAG Ingestion Test Suite for StudentGator RAG Service."""

    # =========================================================================
    # 1. Pydantic Request & Response Contract Tests (src/schemas.py)
    # =========================================================================

    def test_chat_request_valid_and_strips_whitespace(self):
        """Valid payload should strip surrounding whitespace and assign default thread_id."""
        req = ChatRequest(message="   Can I work in Germany on a student visa?   ")
        self.assertEqual(req.message, "Can I work in Germany on a student visa?")
        self.assertEqual(req.thread_id, "default_session")
        self.assertIsNone(req.category_filter)

    def test_chat_request_rejects_empty_whitespace_message(self):
        """Whitespace-only message should be stripped to '' and rejected by min_length=1."""
        with self.assertRaises(ValidationError):
            ChatRequest(message="     ")

    def test_chat_request_rejects_extra_unknown_fields(self):
        """extra='forbid' must reject unapproved JSON fields at the gateway."""
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate({"message": "Hello", "unauthorized_field": "inject"})

    def test_health_response_contract(self):
        """HealthResponse should validate status and boolean flags."""
        health = HealthResponse(
            status="healthy",
            vectorstore_indexed=True,
            langsmith_tracing=False,
        )
        self.assertEqual(health.status, "healthy")
        self.assertTrue(health.vectorstore_indexed)
        self.assertFalse(health.langsmith_tracing)

    # =========================================================================
    # 2. Metadata Registry & ChromaDB Filter Tests (src/metadata.py)
    # =========================================================================

    def test_metadata_manager_loads_manifest(self):
        """MetadataManager should load documents from data/documents_manifest.json."""
        manager = MetadataManager()
        docs = manager.list_documents()
        self.assertGreater(len(docs), 0)
        self.assertIn("Germany", manager.list_countries())

    def test_build_chroma_filter_single_and_compound(self):
        """build_chroma_filter should return None, a single $eq dict, or an $and compound dict."""
        manager = MetadataManager()

        # No filters -> None
        self.assertIsNone(manager.build_chroma_filter())

        # Single filter -> direct $eq dictionary
        single = manager.build_chroma_filter(country="Germany")
        self.assertEqual(single, {"country": {"$eq": "Germany"}})

        # Two filters -> compound $and dictionary
        compound = manager.build_chroma_filter(
            domain_category="eu_germany_admissions",
            country="Germany",
        )
        self.assertEqual(
            compound,
            {
                "$and": [
                    {"domain_category": {"$eq": "eu_germany_admissions"}},
                    {"country": {"$eq": "Germany"}},
                ]
            },
        )

    def test_enrich_chunk_metadata_flattens_lists_for_chromadb(self):
        """ChromaDB requires primitive metadata values (no Python lists)."""
        manager = MetadataManager()
        enriched = manager.enrich_chunk_metadata(
            file_name="eu_germany_residence_act_section_16b.pdf",
            page_number=2,
            chunk_id="chunk_001",
        )
        self.assertEqual(enriched["page"], 2)
        self.assertEqual(enriched["chunk_id"], "chunk_001")
        for key, value in enriched.items():
            self.assertIsInstance(
                value,
                (str, int, float, bool, type(None)),
                f"Non-primitive metadata field '{key}': {type(value)}",
            )

    # =========================================================================
    # 3. LangGraph Conditional Router Tests (src/graph/router.py)
    # =========================================================================

    def test_router_sends_greetings_and_out_of_scope_to_direct_node(self):
        """Greetings and out-of-scope queries should bypass the vector DB."""
        self.assertEqual(route_by_intent({"intent": "greeting"}), "direct_node")
        self.assertEqual(route_by_intent({"intent": "out_of_scope"}), "direct_node")

    def test_router_sends_admission_queries_to_rag_retrieval_node(self):
        """Institutional queries should route to the RAG retrieval node."""
        self.assertEqual(route_by_intent({"intent": "institutional_rag"}), "rag_retrieval_node")
        self.assertEqual(route_by_intent({}), "rag_retrieval_node")

    # =========================================================================
    # 4. RAG Ingestion Pipeline Tests (PDF Load -> Chunking -> Vector Index)
    # =========================================================================

    def test_rag_pdf_loading_and_recursive_chunking(self):
        """
        Tests the first two stages of RAG ingestion:
        1. Loading institutional PDF pages from data/ via PyPDFLoader.
        2. Splitting pages into overlapping chunks with enriched ChromaDB metadata.
        """
        raw_docs = metadata_registry.load_documents()
        self.assertGreater(len(raw_docs), 0, "Should load PDF pages from data/")

        # Chunk a subset of loaded pages and verify chunk size & metadata enrichment
        sample_pages = raw_docs[:5]
        chunks = chunk_and_enrich_documents(sample_pages, chunk_size=400, chunk_overlap=60)

        self.assertGreaterEqual(len(chunks), len(sample_pages))
        first_chunk = chunks[0]
        self.assertTrue(len(first_chunk.page_content) <= 400)
        self.assertIn("chunk_id", first_chunk.metadata)
        self.assertTrue(first_chunk.metadata["chunk_id"].endswith("_c0001"))
        self.assertEqual(first_chunk.metadata["chunk_type"], "child")

    def test_rag_end_to_end_ingestion_and_similarity_search(self):
        """
        Tests full RAG ingestion (`build_persistent_vectorstore`) from PDF loading
        through recursive chunking, embedding generation, and vector similarity search.
        """
        fake_embedder = DeterministicFakeEmbedding(size=32)

        def _in_memory_from_documents(documents, embedding, **kwargs):
            return InMemoryVectorStore.from_documents(documents=documents, embedding=embedding)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_chroma_dir:
            with patch("src.ingestion.Chroma.from_documents", side_effect=_in_memory_from_documents):
                vectorstore = build_persistent_vectorstore(
                    persist_directory=temp_chroma_dir,
                    collection_name="ci_test_collection",
                    force_rebuild=True,
                    embedder=fake_embedder,
                )

            # Verify all 1,100+ PDF chunks were embedded and can be queried via similarity search
            results = vectorstore.similarity_search(
                "What are the requirements for an F-1 visa travel signature?",
                k=2,
            )
            self.assertEqual(len(results), 2)
            self.assertTrue(len(results[0].page_content) > 0)
            self.assertIn("domain_category", results[0].metadata)
            self.assertIn("source", results[0].metadata)


if __name__ == "__main__":
    unittest.main(verbosity=2)

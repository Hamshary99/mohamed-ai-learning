"""
==============================================================================
Week 3 Day 2: Advanced RAG System — Execution & Conceptual Demo
==============================================================================
Student: Mohamed
Course: StudentGator AI Engineering Intensive
Reference Notes: week3/day2/notes/Advanced Rag.md

This script serves as a complete hands-on tutorial and execution workbook for
Advanced Retrieval-Augmented Generation (Advanced RAG). It demonstrates the
five core architectural pillars covered in the lesson:

  1. Hierarchical Chunking (Parent-Child Context Decoupling)
  2. Hybrid Retrieval (Dense Semantic Vector + Sparse Lexical BM25)
  3. Incompatible Score Fusion via Reciprocal Rank Fusion (RRF)
  4. Pre-Retrieval Optimization (Semantic & Statutory Query Expansion)
  5. Two-Stage Re-Ranking (Bi-Encoder Recall -> Cross-Encoder Joint Scoring)
==============================================================================
"""

import os
import sys
from typing import List
from advanced_rag_system import AdvancedRAGSystem


def print_concept_header(step_num: int, title: str, concept_summary: str):
    """Utility to print clean, readable educational section breaks."""
    separator = "=" * 80
    print(f"\n{separator}")
    print(f" STEP {step_num}: {title.upper()}")
    print(separator)
    print(f" Theoretical Mechanics (from Advanced Rag.md):")
    for line in concept_summary.strip().split("\n"):
        print(f"   {line.strip()}")
    print("-" * 80)


def main():
    print("*" * 80)
    print("  STUDENTGATOR AI ENGINEERING: ADVANCED RAG MASTER WORKBENCH")
    print("  Corpus: 5 Official Multi-Jurisdiction Regulatory Handbooks")
    print("          (US DHS OPT, UF F-1, Germany 16b, France VLS-TS, Spain TIE)")
    print("*" * 80)

    system = AdvancedRAGSystem()

    # =========================================================================
    # STEP 1: Multi-Jurisdiction Document Ingestion & Metadata Enrichment
    # =========================================================================
    print_concept_header(
        step_num=1,
        title="Multi-Document Ingestion & Metadata Tagging",
        concept_summary=(
            "Naive flat RAG searches across all departments and countries blindly.\n"
            "By attaching structured metadata (source, domain_category, country) at\n"
            "ingestion time, we enable pre-filtering and exact statutory attribution."
        )
    )
    raw_docs = system.ingest_all_local_pdfs()
    print(f"\n-> Successfully ingested {len(raw_docs)} total document pages.")
    print("-> Attached metadata keys: ['source', 'file_name', 'domain_category', 'source_type']")

    # =========================================================================
    # STEP 2: Hierarchical Parent-Child Chunking
    # =========================================================================
    print_concept_header(
        step_num=2,
        title="Hierarchical Chunking (The Chunk Size Dilemma)",
        concept_summary=(
            "The Chunk Size Dilemma:\n"
            "- Small chunks (200 chars): Sharp vector representations, easy to retrieve, but\n"
            "  starve the LLM of narrative context and qualifying rules.\n"
            "- Large chunks (800+ chars): Rich narrative context for generation, but average out\n"
            "  multiple distinct ideas into a blurry vector, hurting search precision.\n\n"
            "Solution: Decouple Search from Generation.\n"
            "- Child chunks (200 chars) are stored in ChromaDB and BM25 for HIGH PRECISION SEARCH.\n"
            "- Parent chunks (800 chars) are stored in an in-memory docstore for RICH CONTEXT."
        )
    )
    child_chunks = system.create_hierarchical_chunks(
        raw_docs,
        parent_chunk_size=800,
        child_chunk_size=200
    )
    print(f"\n-> DocStore Parent Documents (Rich Generation Context) : {len(system.parent_docstore)}")
    print(f"-> Searchable Child Chunks (Sharp Vector/Lexical Seeds) : {len(child_chunks)}")

    # Educational comparison: Pick a parent with multiple child chunks to clearly illustrate the 800-to-200 char division
    sample_parent_id = None
    sample_children = []
    for p_id in system.parent_docstore:
        children = [c for c in child_chunks if c.metadata.get("parent_id") == p_id]
        if len(children) >= 3:
            sample_parent_id = p_id
            sample_children = children
            break

    if not sample_parent_id:
        sample_parent_id = next(iter(system.parent_docstore))
        sample_children = [c for c in child_chunks if c.metadata.get("parent_id") == sample_parent_id]

    sample_parent = system.parent_docstore[sample_parent_id]

    print(f"\n[Concrete Parent-Child Inspection Example]:")
    print(f"   Parent ID : {sample_parent_id} ({len(sample_parent.page_content)} chars)")
    clean_parent = " ".join(sample_parent.page_content.split())
    print(f"   Parent Preview:\n   \"{clean_parent[:180]}...\"")
    print(f"   Subdivided into {len(sample_children)} child chunks for vector search:")
    for idx, c in enumerate(sample_children[:3], start=1):
        clean_c = " ".join(c.page_content.split())
        print(f"     Child #{idx} [{c.metadata.get('child_id')}] ({len(c.page_content)} chars): \"{clean_c[:90]}...\"")

    # =========================================================================
    # STEP 3: Hybrid Retrieval Index (Chroma Dense + BM25 Sparse + RRF)
    # =========================================================================
    print_concept_header(
        step_num=3,
        title="Hybrid Search Indexing (Dense Semantic + Sparse BM25 Lexical)",
        concept_summary=(
            "Dense Vector (Chroma): Captures high-level intent, synonyms, and paraphrases.\n"
            "Sparse BM25: Guarantees exact matches on technical codes ('Form I-20', '§16b', 'TIE').\n\n"
            "The Scoring Challenge: Cosine similarity produces 0.0 to 1.0, while BM25 produces\n"
            "unbounded scores (e.g. 24.5). Adding them directly causes BM25 to drown out dense vectors.\n"
            "Reciprocal Rank Fusion (RRF) fixes this by combining candidates purely by finish rank\n"
            "using formula: Score(d) = sum( 1 / (60 + rank(d)) )."
        )
    )
    system.initialize_retrieval_index(child_chunks)

    # =========================================================================
    # STEP 4: Pre-Retrieval Optimization: Semantic & Statutory Query Expansion
    # =========================================================================
    print_concept_header(
        step_num=4,
        title="Pre-Retrieval Optimization: Query Expansion",
        concept_summary=(
            "User queries rarely use exact statutory vocabulary (e.g., asking 'how to prove job\n"
            "matches my degree' instead of '8 CFR 214.2(f) direct relationship to major').\n"
            "Query expansion augments the prompt into parallel search variations containing\n"
            "formal administrative acronyms to cast a wide net across the vector index."
        )
    )

    test_queries = [
        "How do I prove my job is directly related to my major for OPT?",
        "Can international students work in Germany while studying?",
        "What are the requirements to validate a student visa in France?",
        "What is the procedure to obtain a TIE card in Spain?"
    ]

    for idx, query in enumerate(test_queries, start=1):
        variations = system.expand_query(query)
        print(f"\n[Test Query {idx}]: \"{query}\"")
        for v_idx, var in enumerate(variations, start=1):
            tag = "Original" if v_idx == 1 else f"Expansion #{v_idx-1}"
            print(f"   [{v_idx}] ({tag}): {var}")

    # =========================================================================
    # STEP 5: Two-Stage Re-Ranking & Grounded Answer Synthesis
    # =========================================================================
    print_concept_header(
        step_num=5,
        title="Two-Stage Re-Ranking (Bi-Encoder Recall -> Cross-Encoder Precision)",
        concept_summary=(
            "Stage 1 (Bi-Encoder / Hybrid): High recall, scans thousands of chunks in milliseconds.\n"
            "Stage 2 (Cross-Encoder / ms-marco-MiniLM): High precision. Feeds concatenated\n"
            "[CLS] Query [SEP] Document [SEP] into a Transformer where every query token attends\n"
            "to every document token. Catches negation, nuanced qualifiers, and legal restrictions.\n\n"
            "Parent Context Resolution: The top-ranked child chunk is mapped back to its parent\n"
            "document in docstore so the generator LLM receives the full surrounding legal text."
        )
    )

    for idx, query in enumerate(test_queries, start=1):
        print(f"\n>>> Running End-to-End Pipeline for Case #{idx}...")
        rag_result = system.generate_answer(query=query, top_k=2, confidence_threshold=1.0)
        
        # Display the fine print diagnostic inspector
        system.print_pipeline_diagnostics(rag_result, show_full_context=False)

    # =========================================================================
    # STEP 6: Stress Testing Out-of-Scope Queries & Confidence Rejection Gate
    # =========================================================================
    print_concept_header(
        step_num=6,
        title="Confidence Scoring Threshold vs Naive Blind Top-K (The Rejection Gate)",
        concept_summary=(
            "Naive RAG Failure Mode:\n"
            "Hardcodes top_k=2 and blindly feeds the top 2 documents to the LLM regardless of relevance.\n"
            "If a user asks about baking pizza or Canadian immigration, naive RAG forces the highest available\n"
            "chunks into the prompt, causing confident hallucinations.\n\n"
            "Production RAG Solution (Scoring Threshold / Cutoff Gate):\n"
            "Sets a confidence cutoff (threshold >= 1.0). If the top Cross-Encoder score is below this cutoff,\n"
            "the pipeline intercepts execution BEFORE calling the LLM and outputs a deterministic refusal."
        )
    )

    out_of_scope_queries = [
        "How do I bake an authentic Neapolitan sourdough pizza at home?",
        "What are the eligibility criteria for a Post-Graduation Work Permit (PGWP) in Canada?",
        "What is the surgical residency matching algorithm for medical students?"
    ]

    for idx, oos_query in enumerate(out_of_scope_queries, start=1):
        print(f"\n>>> Running Stress Test Case #{idx} (Deliberately Out-of-Scope)...")
        rag_result = system.generate_answer(query=oos_query, top_k=2, confidence_threshold=1.0)
        system.print_pipeline_diagnostics(rag_result, show_full_context=False)

    print("\n" + "*" * 80)
    print("  ALL 6 STEPS COMPLETED SUCCESSFULLY!")
    print("  You have tested:")
    print("  1. Ingestion & Structured Metadata Enrichment")
    print("  2. Parent-Child Decoupled Chunking (200 char child / 800 char parent)")
    print("  3. Dense ChromaDB + Sparse BM25 Hybrid Retrieval with RRF")
    print("  4. Pre-Retrieval Multi-Query Expansion")
    print("  5. Cross-Encoder Joint Re-Ranking & Grounded Multi-Jurisdiction Synthesis")
    print("  6. Confidence Rejection Gate (Blocking Unrelated & Cross-Border Out-of-Scope Queries)")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()

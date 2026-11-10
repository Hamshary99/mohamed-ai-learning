# Engineering Advanced Retrieval-Augmented Generation: Practical Python Architecture for Production Systems

## 1. Architectural Framework: The Transition from Naive to Advanced RAG

Standard naive Retrieval-Augmented Generation relies on a linear, single-pass pipeline: an incoming user query is embedded into a dense coordinate vector, matched against pre-indexed text chunks in a vector database via cosine similarity, and the top candidate chunks are concatenated directly into an LLM prompt. While this pattern proves effective for basic question-answering over clean, narrative documents, it exhibits structural failure modes when deployed against production enterprise datasets:

1. **The Vocabulary Mismatch and Keyword Blind Spot**: Dense bi-encoders map high-level semantic themes into geometric coordinate space. They understand that *housing* relates conceptually to *dormitories*, but they frequently miss exact alphanumeric entities, error codes, form numbers, or statutory acronyms (such as *Form I-20*, *SG-402*, or *SEVIS ID*) because these rare tokens lack rich semantic neighborhood context.
2. **The Chunk-Size Dilemma**: If source documents are split into narrow chunks (such as single sentences), vector similarity matching is sharp and precise, but the generation model lacks the contextual surrounding paragraphs needed to resolve references and answer comprehensively. Conversely, if documents are split into wide chunks (such as 1,500 tokens), the embedding vector averages across multiple topics, diluting specific facts and degrading retrieval precision.
3. **Query Ambiguity and Multi-Part Questions**: Real-world user queries are rarely optimal search vectors. Users pose compound, comparative questions or use colloquial phrasing that places the query vector outside the neighborhood of the relevant documentation.
4. **Context Dilution and In-Context Distraction**: Feeding unranked or loosely related candidate chunks into an LLM context window increases token costs and triggers the *lost-in-the-middle* phenomenon, where generative models overlook key facts placed between unrelated context fragments.

To resolve these limitations, modern production systems implement an advanced three-tier retrieval pipeline:

* **Pre-Retrieval & Ingestion**: Ingesting raw external documents, automated metadata extraction, hierarchical recursive character splitting, and query transformation (expansion and multi-query decomposition).
* **Multi-Stream Retrieval**: Parallel dense semantic vector search (ChromaDB) and sparse lexical search (BM25), constrained by relational metadata pre-filters.
* **Post-Retrieval Optimization**: Reciprocal Rank Fusion (RRF) with weighted ranking strategies, followed by cross-encoder re-ranking for joint self-attention scoring, truncating candidates to the highest-precision subset.

This engineering guide provides a complete architectural and programmatic implementation in Python using LangChain, ChromaDB, and SentenceTransformers, applied to live online sources and institutional knowledge bases.

---

## 2. Ingesting Real-World Online Documents and Automated Metadata Architecture

A foundational question in production RAG systems concerns document ingestion: should systems rely on hardcoded text strings, and must engineers manually author metadata dictionaries for every indexed chunk? In production engineering, the answer is unequivocally no. Systems ingest raw documents from networks or file systems and generate metadata programmatically.

### Recommended Public Online Sources for RAG Benchmarking

When testing and evaluating advanced retrieval architectures, selecting domain-relevant corpora provides realistic stress-testing for acronyms, hierarchical structures, and cross-document reasoning. Three exemplary categories of public online sources serve this purpose:

1. **Institutional International Student Regulations (StudentGator Domain)**: Public university international student compliance guides—such as the University of Florida International Center F-1 Regulations handbook—contain exact statutory terms (*Form I-20*, *SEVIS*, *OPT*, *CPT*, *DSO endorsement*), strict timelines, and financial penalty clauses. These documents effectively stress-test hybrid search and metadata scoping.
2. **Federal Immigration and Labor Directives**: Official policy guidance from the Department of Homeland Security (DHS) and U.S. Citizenship and Immigration Services (USCIS) regarding student employment authorization and visa status maintenance provide dense regulatory language ideal for testing cross-encoder re-ranking against subtle phrasing nuances.
3. **Machine Learning Research Manuscripts (arXiv)**: Seminal research publications—such as the original Retrieval-Augmented Generation paper by Lewis et al. (arXiv:2005.11401) or the Attention Is All You Need paper by Vaswani et al. (arXiv:1706.03762)—present complex multi-section layouts, mathematical formulations, and dense bibliographies that test recursive chunking and table handling.

### How Metadata is Created: Four Production Strategies

Metadata serves as the relational scaffolding that enables pre-retrieval filtering, domain routing, and source attribution. Engineers do not write this metadata by hand. Instead, production architectures leverage four systematic generation mechanisms:

| Metadata Generation Strategy | Operational Mechanism | Typical Fields Extracted | Implementation Complexity |
| :--- | :--- | :--- | :--- |
| **1. Intrinsic Loader Extraction** | Parsers extract embedded container properties automatically during binary file decoding. | `source`, `page_number`, `total_pages`, `creation_date`, `author` | Zero additional code; provided natively by document loaders. |
| **2. Deterministic Rule Extraction** | String manipulation and regular expressions parse directory structures, URL paths, and file naming conventions. | `department`, `document_type`, `fiscal_year`, `access_tier` | Low; executed via path splitting and regex pattern matching. |
| **3. Structural Header Parsing** | Text parsers identify markdown headers (`#`, `##`) or PDF table-of-contents trees, propagating section names down to child chunks. | `chapter_title`, `section_header`, `subsection_hierarchy` | Moderate; leverages structural splitters (e.g., `MarkdownHeaderTextSplitter`). |
| **4. LLM-Assisted Automated Extraction** | A lightweight, fast language model processes each raw chunk during ingestion to output structured JSON schemas. | `summary`, `domain_keywords`, `entity_names`, `target_audience` | High; incurs minor token cost and ingestion latency, but yields rich semantic filters. |

#### Strategy 1: Intrinsic and Deterministic Ingestion

When a file loader processes a document, it automatically creates a dictionary of intrinsic file attributes. Developers supplement these with deterministic rules derived from the file origin:

```python
import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def fetch_and_index_online_pdf(url: str, cache_dir: str = "data_cache"):
    """
    Downloads an online PDF with local filesystem caching, extracts intrinsic page metadata,
    and programmatically appends deterministic domain tags.
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_name = url.split("/")[-1] or "document.pdf"
    cached_path = os.path.join(cache_dir, file_name)

    # 1. Download only if not already cached locally
    if not os.path.exists(cached_path):
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        with open(cached_path, "wb") as f:
            f.write(response.content)

    # 2. Extract text and intrinsic metadata via PyPDFLoader
    loader = PyPDFLoader(cached_path)
    raw_pages = loader.load()

    # 3. Deterministic metadata enrichment derived from path and file naming
    for page_idx, page in enumerate(raw_pages):
        page.metadata["source_url"] = url
        page.metadata["document_name"] = file_name
        page.metadata["ingestion_source"] = "institutional_web_crawler"
        # Deterministic domain tagging based on filename patterns
        if "visa" in file_name.lower() or "international" in file_name.lower():
            page.metadata["department"] = "immigration_services"
        elif "housing" in file_name.lower():
            page.metadata["department"] = "residential_life"
        else:
            page.metadata["department"] = "general_academic"

    # 4. Hierarchical recursive splitting preserving enriched metadata
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunked_documents = splitter.split_documents(raw_pages)
    
    return chunked_documents
```

#### Strategy 2: LLM-Assisted Metadata Extraction

For high-value enterprise repositories where queries require fine-grained relational filtering, an asynchronous Small Language Model (such as `gpt-4o-mini` or an open-weights SLM) can inspect each chunk during the ingestion pipeline to generate structured semantic metadata tags:

```python
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class ChunkEnrichmentSchema(BaseModel):
    category: str = Field(description="Primary administrative category: immigration, finance, housing, or academic")
    key_entities: list[str] = Field(description="Specific statutory forms, official codes, or proper names mentioned")
    confidentiality: str = Field(description="Access level: public, internal_student, or administrative_only")

enrichment_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract structured regulatory metadata attributes for the provided document excerpt."),
    ("human", "{chunk_text}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
structured_extractor = enrichment_prompt | llm.with_structured_output(ChunkEnrichmentSchema)
```

---

## 3. Stage 1: Pre-Retrieval Optimization (Hierarchical Ingestion and Context Decoupling)

### The Chunking Trade-Off

Document ingestion requires a balance between text coherence and vector specificity. Fixed-size chunking divides text strictly by character or token limits, which risks cutting sentences, equations, or lists across arbitrary boundaries. In contrast, recursive character chunking uses a hierarchical list of separators:

```python
separators = ["\n\n", "\n", ". ", " ", ""]
```

The algorithm evaluates the largest structural boundary first (paragraphs), splitting only when a unit exceeds the target size, and falls back to sentences or words only when necessary. This preserves complete thoughts within a single chunk.

To implement hierarchical context enrichment (Parent-Child indexing), documents are split at two distinct resolutions:
1. **Child Chunks (150–200 characters)**: Embedded into vector coordinates to optimize retrieval sensitivity.
2. **Parent Chunks (600–1,000 characters)**: Preserved in an auxiliary document store and linked via metadata IDs so that when a child chunk matches a query, the complete parent context is passed to the generation model.

In an alternate pattern, known as Sentence-Window retrieval, individual sentences are embedded in the vector store while their metadata links to an adjacent window of $k$ neighboring sentences before and after, dynamically expanding context at retrieval time.

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)

def create_hierarchical_chunks(base_docs: list[Document]) -> list[Document]:
    parent_docs = parent_splitter.split_documents(base_docs)
    child_docs = []

    for parent_id, p_doc in enumerate(parent_docs):
        p_doc.metadata["parent_id"] = f"parent_node_{parent_id}"
        c_splits = child_splitter.split_documents([p_doc])
        for child_id, c_split in enumerate(c_splits):
            c_split.metadata["parent_id"] = f"parent_node_{parent_id}"
            c_split.metadata["child_id"] = f"child_node_{parent_id}_{child_id}"
            child_docs.append(c_split)
            
    return child_docs
```

---

## 4. Stage 2: Query Transformation (Expansion and Decomposition)

A primary failure mode in vector search is relying on the user's raw phrasing. If an international student queries *"Can I travel home during spring break?"*, a pure vector search over formal regulatory text might fail because the document uses terms like *travel endorsement*, *Form I-20*, and *F-1 re-entry requirements*.

### Query Expansion Architecture

Query expansion employs an LLM or Small Language Model (SLM) to generate three to five alternative phrasings containing domain-specific synonyms. Each variation executes across the retrieval pipeline in parallel, and results are merged to maximize recall:

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import LineListOutputParser

expansion_template = """You are an expert AI retrieval query optimizer for academic and immigration records.
Given the following user question, generate exactly 3 to 4 alternative search queries.
Each variation must incorporate formal institutional synonyms, relevant administrative acronyms, or regulatory phrasing.
Output ONLY the alternative queries separated by newlines. Do not include numbered bullets, preambles, or conversational text.

User Question: {question}
"""

class QueryListParser(LineListOutputParser):
    def parse(self, text: str) -> list[str]:
        return [line.strip() for line in text.strip().split("\n") if line.strip() and not line.strip().startswith("#")]

expansion_prompt = PromptTemplate(input_variables=["question"], template=expansion_template)
```

### Multi-Query Decomposition

When users ask compound or comparative questions—such as *"What are the refund consequences of dropping classes compared to canceling a housing contract?"*—a single vector represents a blurred composite of both concepts. Multi-query decomposition parses the prompt into independent, atomic sub-queries:

1. **Sub-Query 1**: *"Student housing contract SG-402 cancellation penalty and deadlines"*
2. **Sub-Query 2**: *"Academic tuition drop schedule and refund percentage deadlines"*

Both sub-queries execute concurrently against the retrieval pipeline, and their candidate pools are unified before generation.

---

## 5. Stage 3: Multi-Stream Hybrid Retrieval (Dense Semantic + Sparse BM25)

Dense semantic embeddings and sparse lexical matching address distinct and complementary search requirements:

| Dimension | Dense Semantic Retrieval (Bi-Encoder) | Sparse Lexical Retrieval (BM25) |
| :--- | :--- | :--- |
| **Matching Mechanism** | Cosine distance across continuous vector spaces | Term frequency–inverse document frequency with length normalization |
| **Strengths** | Captures high-level intent, conceptual abstractions, and natural language synonyms | Exact matches for acronyms, specific form codes (e.g., *SG-402*), and proper nouns |
| **Weaknesses** | Struggles with rare tokens, numerical codes, and exact string matching | Completely misses synonyms (e.g., *dormitory* vs. *housing*) |
| **Computational Mode** | Pre-computed embeddings evaluated via vector index (HNSW / Flat) | Inverted index with low computational overhead |

### Implementing Hybrid Search with LangChain

The `EnsembleRetriever` coordinates dense vector search and sparse BM25 search in parallel, merging candidate lists using Reciprocal Rank Fusion:

```python
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def build_hybrid_retriever(documents: list[Document], embedder):
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedder)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 6

    # Weighted Ensemble: 60% Dense Semantic (Concepts), 40% Sparse Lexical (Acronyms/Codes)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )
    return hybrid_retriever
```

---

## 6. Stage 4: Post-Retrieval Optimization (Ensemble RRF and Cross-Encoder Re-Ranking)

### The Score Disparity Problem and Reciprocal Rank Fusion

When merging candidate lists from BM25 and vector search, their raw scores cannot be combined directly. BM25 generates unbounded positive values (often ranging from 3.0 to 40.0+), whereas cosine similarity outputs normalized floats between -1.0 and 1.0. Directly summing these scores allows BM25 to overpower semantic similarity.

Reciprocal Rank Fusion (RRF) bypasses raw score disparities by evaluating documents strictly according to their rank positions:

$$\text{RRF Score}(d) = \sum_{m \in M} w_m \cdot \frac{1}{k + r_m(d)}$$

Where:
* $M$ is the set of retrievers (dense vector search and BM25).
* $r_m(d)$ is the 1-based rank position of document $d$ within retriever $m$.
* $k$ is a rank smoothing constant (standard default: 60) that prevents top-ranked outliers on a single list from dominating items that placed consistently well across multiple retrievers.
* $w_m$ is the assigned retriever weight.

### Two-Stage Re-Ranking: Bi-Encoder vs. Cross-Encoder

While a Bi-Encoder scans millions of stored vectors in milliseconds, it cannot model token-by-token interactions between the query and candidate documents. It compresses an entire text chunk into a single vector point, missing fine-grained relationships and negations.

A Cross-Encoder takes the query and a document chunk together as a single concatenated input sequence:

$$\text{Input} = [\text{CLS}, \text{Query Tokens}, \text{SEP}, \text{Document Tokens}, \text{EOS}]$$

The model runs full self-attention across every word in the query and every word in the document simultaneously, outputting a precise relevance score between 0 and 1. Because running full cross-attention across thousands of documents at query time is computationally prohibitive, production architectures use a two-stage pipeline:

1. **Stage 1 (Coarse Recall)**: A fast Bi-Encoder and BM25 hybrid search scans the database to retrieve a candidate pool of 20 to 50 documents.
2. **Stage 2 (Fine Precision)**: A Cross-Encoder evaluates the candidate pool with joint self-attention, re-scoring each chunk to yield the top 2 to 5 highest-relevance contexts for prompt injection.

---

## 7. Production Implementation: End-to-End Pipeline with Caching, Parent-Child, and Two-Stage Re-Ranking

The following complete Python script implements the full Day 2 architecture: online PDF acquisition with local disk caching, automated metadata extraction, hierarchical Parent-Child splitting, ChromaDB indexing with metadata filtering, BM25 retrieval, weighted ensemble fusion, Cross-Encoder re-ranking, and answer synthesis with citations.

```python
"""
StudentGator Production Advanced RAG Pipeline
Features: Online PDF Ingestion, Automated Metadata Enrichment,
          Hierarchical Parent-Child Context Resolution,
          Hybrid Search (ChromaDB + BM25) with Metadata Filtering,
          Cross-Encoder Two-Stage Re-Ranking, and Answer Generation.
"""

import os
import requests
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Production Embeddings with graceful offline/lightweight fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    DEFAULT_EMBEDDER = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception:
    from langchain_community.embeddings import FakeEmbeddings
    DEFAULT_EMBEDDER = FakeEmbeddings(size=384)


class AdvancedRAGSystem:
    def __init__(self, cache_dir: str = "data_cache", embedder=None, llm=None):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.embedder = embedder or DEFAULT_EMBEDDER
        # Lightweight, high-accuracy cross-encoder for production re-ranking
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.llm = llm

        # Document Store for hierarchical Parent-Child context resolution
        self.parent_docstore: Dict[str, Document] = {}

        # Search components
        self.vectorstore: Optional[Chroma] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None

    def ingest_online_document(self, url: str, domain_tag: str) -> List[Document]:
        """Downloads external PDF, extracts intrinsic metadata, and applies domain tags."""
        file_name = url.split("/")[-1] or "document.pdf"
        cached_file_path = os.path.join(self.cache_dir, file_name)

        if not os.path.exists(cached_file_path):
            print(f"Acquiring remote resource: {url}...")
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            response.raise_for_status()
            with open(cached_file_path, "wb") as f:
                f.write(response.content)
            print(f"Cached successfully to {cached_file_path}")
        else:
            print(f"Utilizing cached resource: {cached_file_path}")

        # Ingest and parse pages
        loader = PyPDFLoader(cached_file_path)
        pages = loader.load()

        # Programmatic metadata enrichment
        for page in pages:
            page.metadata["source_url"] = url
            page.metadata["file_name"] = file_name
            page.metadata["domain_category"] = domain_tag

        return pages

    def create_hierarchical_chunks(
        self,
        base_docs: List[Document],
        parent_chunk_size: int = 800,
        child_chunk_size: int = 200,
    ) -> List[Document]:
        """
        Splits documents into large Parent Chunks (for generation context)
        and small Child Chunks (for vector/lexical retrieval sensitivity).
        """
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=30,
            separators=["\n\n", "\n", ". ", " "]
        )

        parent_docs = parent_splitter.split_documents(base_docs)
        child_docs: List[Document] = []

        for p_idx, p_doc in enumerate(parent_docs):
            parent_id = f"parent_{p_doc.metadata.get('domain_category', 'general')}_{p_idx}"
            p_doc.metadata["parent_id"] = parent_id
            self.parent_docstore[parent_id] = p_doc

            splits = child_splitter.split_documents([p_doc])
            for c_idx, c_split in enumerate(splits):
                c_split.metadata["parent_id"] = parent_id
                c_split.metadata["child_id"] = f"{parent_id}_c{c_idx}"
                child_docs.append(c_split)

        return child_docs

    def initialize_retrieval_index(self, documents: List[Document]):
        """Builds ChromaDB vector store and BM25 sparse index into an EnsembleRetriever."""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder,
            collection_name="advanced_rag_production"
        )
        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})

        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 8

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, self.bm25_retriever],
            weights=[0.6, 0.4]
        )

    def expand_query(self, query: str) -> List[str]:
        """
        Generates alternative search queries containing formal domain terminology.
        If an LLM is configured, uses prompt generation; otherwise returns deterministic variants.
        """
        if self.llm:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Generate 2 alternative search queries for student services RAG using formal terms. Return only queries separated by newlines."),
                ("human", "{query}")
            ])
            chain = prompt | self.llm | StrOutputParser()
            try:
                result = chain.invoke({"query": query})
                variations = [line.strip() for line in result.split("\n") if line.strip()]
                return [query] + variations[:2]
            except Exception:
                pass

        # Deterministic domain synonyms fallback
        expanded = [query]
        if "housing" in query.lower() or "contract" in query.lower():
            expanded.append(f"{query} residence agreement cancellation fee SG-402")
        if "visa" in query.lower() or "travel" in query.lower():
            expanded.append(f"{query} Form I-20 travel signature endorsement F-1")
        return expanded

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None,
        resolve_parent: bool = True
    ) -> List[Document]:
        """
        Executes:
        1. Multi-query retrieval over Ensemble (Chroma + BM25)
        2. Relational metadata filtering
        3. Cross-Encoder joint self-attention re-scoring
        4. Optional Parent-Child hierarchical context resolution
        """
        if not self.ensemble_retriever or not self.vectorstore:
            raise RuntimeError("Index not initialized. Run initialize_retrieval_index() first.")

        # Step 1: Multi-query candidate expansion
        query_variations = self.expand_query(query)
        candidate_pool: List[Document] = []
        seen_texts = set()

        for q_var in query_variations:
            # Dense search with optional relational metadata filter
            if metadata_filter:
                dense_hits = self.vectorstore.similarity_search(q_var, k=6, filter=metadata_filter)
            else:
                dense_hits = self.vectorstore.similarity_search(q_var, k=6)

            # Sparse lexical BM25 search
            bm25_hits = self.bm25_retriever.invoke(q_var)
            if metadata_filter:
                bm25_hits = [
                    d for d in bm25_hits
                    if all(d.metadata.get(k) == v for k, v in metadata_filter.items())
                ]

            for doc in (dense_hits + bm25_hits):
                if doc.page_content not in seen_texts:
                    seen_texts.add(doc.page_content)
                    candidate_pool.append(doc)

        if not candidate_pool:
            return []

        # Step 2: Cross-attention joint re-scoring
        query_doc_pairs = [[query, doc.page_content] for doc in candidate_pool]
        relevance_scores = self.reranker.predict(query_doc_pairs)

        for doc, score in zip(candidate_pool, relevance_scores):
            doc.metadata["cross_encoder_score"] = float(score)

        # Sort descending by re-rank score
        ranked_candidates = sorted(
            candidate_pool,
            key=lambda d: d.metadata["cross_encoder_score"],
            reverse=True
        )

        # Step 3: Resolve to parent context (Parent-Child pattern)
        final_docs: List[Document] = []
        seen_parents = set()

        for doc in ranked_candidates:
            parent_id = doc.metadata.get("parent_id")
            if resolve_parent and parent_id and parent_id in self.parent_docstore:
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    parent_doc = self.parent_docstore[parent_id]
                    parent_doc.metadata["cross_encoder_score"] = doc.metadata["cross_encoder_score"]
                    final_docs.append(parent_doc)
            else:
                final_docs.append(doc)

            if len(final_docs) >= top_k:
                break

        return final_docs

    def generate_answer(
        self,
        query: str,
        top_k: int = 2,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete RAG pipeline: retrieves reranked parent contexts,
        constructs cited prompt, and synthesizes final answer.
        """
        retrieved_docs = self.retrieve_and_rerank(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            resolve_parent=True
        )

        # Format context and citations
        context_blocks = []
        sources = []
        for i, doc in enumerate(retrieved_docs, start=1):
            src = doc.metadata.get("source", doc.metadata.get("file_name", "Document"))
            page = doc.metadata.get("page", 1)
            score = doc.metadata.get("cross_encoder_score", 0.0)
            sources.append({"source": src, "page": page, "score": score})
            context_blocks.append(f"[Document {i} | Source: {src}, Page {page}]\n{doc.page_content}")

        context_str = "\n\n".join(context_blocks)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant for StudentGator student services. Answer the question using ONLY the provided context excerpts. Cite policy codes, form names, and specific rules. If the context does not contain the answer, state that clearly.\n\nContext:\n{context}"),
            ("human", "{question}")
        ])

        if self.llm:
            chain = prompt_template | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context_str, "question": query})
        else:
            # Deterministic fallback response when running locally without active LLM API keys
            answer = (
                f"[Synthesis based on {len(retrieved_docs)} verified sources]:\n"
                f"{retrieved_docs[0].page_content if retrieved_docs else 'No matching policy found.'}"
            )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "context_used": context_str
        }


# =========================================================================
# Execution Demonstration
# =========================================================================
if __name__ == "__main__":
    system = AdvancedRAGSystem()

    # Raw institutional policy documents with enriched metadata
    raw_policies = [
        Document(
            page_content="Form I-20 Travel Signature Policy: Active international F-1 students require an unexpired travel endorsement on page 2 of Form I-20 prior to leaving the United States. Travel signatures remain valid for 12 months for enrolled degree-seeking students, but only for 6 months for alumni engaged in post-completion Optional Practical Training (OPT). Incomplete endorsements may lead to Form I-515A issuance or entry refusal at U.S. border checkpoints.",
            metadata={"source": "visa_policy.pdf", "page": 2, "domain_category": "immigration"}
        ),
        Document(
            page_content="Student Housing Agreement Cancellation (Form SG-402): Terminating an on-campus residential housing contract after August 15 incurs a mandatory 50% tuition and housing assessment penalty. Exception requests require documented visa denial from a U.S. Embassy or formal military induction orders submitted through the Housing Portal.",
            metadata={"source": "housing_policy.pdf", "page": 5, "domain_category": "residential_life"}
        ),
        Document(
            page_content="Tuition Refund and Course Drop Schedule: Full tuition refunds (100%) are issued for drops occurring through Day 5 of the academic semester. Course withdrawals between weeks 2 and 4 receive a 50% bursar credit. International student orientation and SEVIS administrative fees ($350) are non-refundable under all circumstances.",
            metadata={"source": "finance_policy.pdf", "page": 1, "domain_category": "bursar"}
        )
    ]

    print("Step 1: Creating Hierarchical Parent-Child Chunks...")
    child_chunks = system.create_hierarchical_chunks(raw_policies, parent_chunk_size=300, child_chunk_size=100)
    print(f"Created {len(system.parent_docstore)} parent contexts and {len(child_chunks)} searchable child chunks.")

    print("\nStep 2: Initializing ChromaDB and BM25 Ensemble Retriever...")
    system.initialize_retrieval_index(child_chunks)

    test_query = "What happens if I cancel my housing agreement after August 15?"
    print(f"\nStep 3: Running Complete Advanced RAG Pipeline for query: '{test_query}'")

    # Run with metadata scoping to residential_life
    result = system.generate_answer(
        query=test_query,
        top_k=1,
        metadata_filter={"domain_category": "residential_life"}
    )

    print("\n=== GENERATED RESPONSE ===")
    print(result["answer"])
    print("\n=== VERIFIED SOURCES ===")
    for s in result["sources"]:
        print(f"Source: {s['source']} (Page {s['page']}) | Cross-Encoder Score: {s['score']:.4f}")
```

---

## 8. Technical Checkpoint Review: Defending Architectural Choices to Mentors

During Checkpoint Meeting #3, technical mentors evaluate both code implementation and the underlying architectural rationale. The following questions and answers address key evaluation criteria:

### Question 1: In production RAG, do developers manually create document metadata?
> **Answer**: No. Developers do not write manual metadata dictionaries for corpus documents. Document loaders (such as `PyPDFLoader`) automatically extract intrinsic metadata—including source URLs, page numbers, and total page counts—directly from container files. Furthermore, ingestion scripts use deterministic rules (parsing directory names, URLs, and filenames) and structural heading splitters to enrich chunks with domain categories and section titles. In advanced systems, lightweight SLMs are deployed asynchronously during ingestion to extract structured entities and summaries automatically.

### Question 2: Why pair BM25 with ChromaDB instead of relying solely on vector embeddings?
> **Answer**: Dense vector models excel at semantic comprehension and synonym recognition, but they treat rare alphanumeric identifiers (such as *SG-402*, *I-20*, or policy codes) as low-frequency tokens, often causing them to miss exact form matches. BM25 indexes term frequencies and normalizes for document length, guaranteeing exact-string recall. Combining them in an ensemble balances conceptual coverage with keyword precision.

### Question 3: Why can't we sum or average BM25 scores and cosine similarity scores directly?
> **Answer**: BM25 outputs unbounded positive numbers determined by term frequency and document length, while cosine similarity yields bounded values between -1.0 and 1.0. Directly summing these mismatched scales allows BM25 to overpower the vector score. Reciprocal Rank Fusion (RRF) resolves this by evaluating candidates strictly by their rank positions rather than arbitrary raw scores.

### Question 4: What is the architectural difference between a Bi-Encoder and a Cross-Encoder, and why not use the Cross-Encoder for the entire database?
> **Answer**: A Bi-Encoder encodes queries and documents into independent vector embeddings ahead of time, allowing Approximate Nearest Neighbor (ANN) indexing to scan millions of documents in milliseconds. However, it cannot model word-by-word interactions between the query and text. A Cross-Encoder feeds the query and document together through full self-attention, evaluating token interactions for higher accuracy. Running full self-attention across an entire database at query time introduces unacceptable latency, which is why production systems use a two-stage pattern: a fast Bi-Encoder retrieves the top 20–50 candidates, and a Cross-Encoder re-ranks them down to the top 2–5 chunks.

### Question 5: How does Parent-Child retrieval resolve the chunk-size trade-off?
> **Answer**: Narrow chunks (100–200 tokens) yield sharp, uncluttered vector embeddings that match queries reliably, but lack the surrounding narrative context needed for generation. Wide chunks (800+ tokens) provide comprehensive context for the LLM, but their vector embeddings become diluted averages that degrade retrieval precision. Parent-Child retrieval decouples search from generation by indexing small child chunks for vector matching, then resolving their parent IDs to pass the broader parent passage to the generation model.

### Question 6: How does metadata pre-filtering protect both retrieval speed and accuracy in high-volume RAG systems?
> **Answer**: Metadata pre-filtering applies relational constraints (e.g. `domain_category == 'residential_life'` or `fiscal_year == 2026`) *before* executing the geometric vector similarity search. In production datasets with millions of vectors, searching the global index introduces cross-domain noise (e.g. visa policies diluting housing policies) and increases ANN search latency. Pre-filtering bounds the search space strictly to the relevant document subset, eliminating false positives and reducing query execution times.

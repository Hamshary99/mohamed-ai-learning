"""
Week 3 Day 2: Advanced RAG System Module
=========================================
Implements the core AdvancedRAGSystem class:
- Document Ingestion (Local PDF & Online Caching)
- Hierarchical Parent-Child Chunking
- Hybrid Retrieval (Chroma + BM25)
- Cross-Encoder Re-Ranking
- Grounded Answer Generation
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional

# LangChain Core & Text Splitters
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Vector Store, Embeddings & Sparse Search
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain_community.retrievers import EnsembleRetriever

# Embeddings & Cross-Encoder Reranker
from sentence_transformers import CrossEncoder

# Production Embeddings with graceful offline/lightweight fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    DEFAULT_EMBEDDER = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception:
    from langchain_community.embeddings import FakeEmbeddings
    DEFAULT_EMBEDDER = FakeEmbeddings(size=384)


class AdvancedRAGSystem:
    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_dir: str = "data_cache",
        embedder=None,
        llm=None
    ):
        """
        Initializes the Advanced RAG pipeline.
        :param data_dir: Directory containing local reference PDFs.
        :param cache_dir: Directory for caching remotely downloaded PDFs.
        :param embedder: Dense vector embedding model (defaults to all-MiniLM-L6-v2).
        :param llm: Chat model for query expansion and answer generation.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir or os.path.join(current_dir, "..", "data")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.embedder = embedder or DEFAULT_EMBEDDER
        # Lightweight, high-accuracy cross-encoder for production re-ranking
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.llm = llm

        # Document Store for hierarchical Parent-Child context resolution
        self.parent_docstore: Dict[str, Document] = {}

        # Search components (to be initialized via initialize_retrieval_index)
        self.vectorstore: Optional[Chroma] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None

    def ingest_local_pdf(self, file_path_or_name: str, domain_tag: str) -> List[Document]:
        """
        Loads a local PDF document, parses its pages with PyPDFLoader, and programmatically
        enriches each page document with deterministic domain metadata attributes.
        """
        if os.path.isabs(file_path_or_name) or os.path.exists(file_path_or_name):
            target_path = file_path_or_name
        else:
            target_path = os.path.join(self.data_dir, file_path_or_name)

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"PDF not found at: {target_path}")

        file_name = os.path.basename(target_path)
        loader = PyPDFLoader(target_path)
        pages = loader.load()

        # Deterministic metadata enrichment
        for page in pages:
            page.metadata["source"] = file_name
            page.metadata["file_name"] = file_name
            page.metadata["domain_category"] = domain_tag
            page.metadata["source_type"] = "local_pdf"

        print(f"Ingested '{file_name}': {len(pages)} pages loaded with domain '{domain_tag}'.")
        return pages

    def ingest_all_local_pdfs(self) -> List[Document]:
        """
        Loads all PDF documents from self.data_dir using metadata from documents_manifest.json
        if available, or automatically tagging files based on their naming conventions.
        """
        all_docs: List[Document] = []
        manifest_path = os.path.join(self.data_dir, "documents_manifest.json")
        manifest_map = {}

        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    for item in manifest_data:
                        manifest_map[item["file_name"]] = item.get("domain_category", "general")
            except Exception as e:
                print(f"Warning: Could not parse documents_manifest.json: {e}")

        pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            domain = manifest_map.get(pdf_file, "general")
            docs = self.ingest_local_pdf(pdf_file, domain_tag=domain)
            all_docs.extend(docs)

        print(f"Ingested all {len(pdf_files)} local PDFs ({len(all_docs)} total pages loaded).")
        return all_docs

    def ingest_online_document(self, url: str, domain_tag: str) -> List[Document]:
        """
        Downloads an external PDF with local caching, extracts page text,
        and enriches each page with deterministic metadata attributes.
        """
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

        loader = PyPDFLoader(cached_file_path)
        pages = loader.load()

        for page in pages:
            page.metadata["source_url"] = url
            page.metadata["file_name"] = file_name
            page.metadata["domain_category"] = domain_tag
            page.metadata["source_type"] = "online_pdf"

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
        # 1. Define two splitters with distinct resolution levels:
        #    - Parent splitter (800 chars): Keeps broad paragraphs intact for LLM synthesis.
        #    - Child splitter (200 chars): Produces sharp, focused vectors for retrieval accuracy.
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

        # 2. First pass: Split raw pages into comprehensive parent chunks
        parent_docs = parent_splitter.split_documents(base_docs)
        child_docs: List[Document] = []

        # 3. Second pass: Register parents in memory docstore and carve out searchable child chunks
        for p_idx, p_doc in enumerate(parent_docs):
            # Assign unique parent_id and save to docstore dictionary
            parent_id = f"parent_{p_doc.metadata.get('domain_category', 'general')}_{p_idx}"
            p_doc.metadata["parent_id"] = parent_id
            self.parent_docstore[parent_id] = p_doc

            # Sub-split the parent chunk into micro child chunks
            splits = child_splitter.split_documents([p_doc])
            for c_idx, c_split in enumerate(splits):
                # Tag child with parent_id so the retriever can fetch parent context later
                c_split.metadata["parent_id"] = parent_id
                c_split.metadata["child_id"] = f"{parent_id}_c{c_idx}"
                child_docs.append(c_split)

        # Return child chunks to be embedded in ChromaDB & BM25
        return child_docs

    def initialize_retrieval_index(self, documents: List[Document]):
        """
        Builds ChromaDB dense vector store and BM25 sparse lexical index,
        unifying them into a weighted EnsembleRetriever (60% Dense / 40% Sparse).
        """
        # Step 1: Dense Vector Index (ChromaDB)
        # Embeds each child chunk using all-MiniLM-L6-v2 for semantic comprehension & synonyms
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder,
            collection_name="advanced_rag_production"
        )
        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})

        # Step 2: Sparse Lexical Index (BM25)
        # Inverted index based on TF-IDF with length normalization for exact acronyms/codes (e.g. Form I-20, SG-402, TIE)
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 8

        # Step 3: Ensemble Fusion via Reciprocal Rank Fusion (RRF)
        # Combines dense and sparse results by ranking positions without raw score mismatch
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, self.bm25_retriever],
            weights=[0.6, 0.4]  # 60% semantic concepts, 40% exact keyword safeguards
        )
        print(f"Retrieval Index initialized with {len(documents)} searchable chunks (ChromaDB + BM25).")

    def expand_query(self, query: str) -> List[str]:
        """
        Augments a raw user query with domain-specific statutory synonyms and official codes.
        If an LLM is configured, uses prompt generation; otherwise applies deterministic rules
        aligned with the local documents in coding/data/.
        """
        # Strategy A: LLM-Assisted Semantic Query Expansion
        if self.llm:
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are an AI query expansion optimizer for international student affairs.\n"
                    "Generate 3 to 5 semantically diverse search queries for the user's question.\n"
                    "Include formal administrative acronyms, statutory terminology, and regulatory phrasing.\n"
                    "Return ONLY the alternative queries separated by newlines, with no numbering, bullets, or preambles."
                ),
                ("human", "{query}")
            ])
            chain = prompt | self.llm | StrOutputParser()
            try:
                result = chain.invoke({"query": query})
                variations = [
                    line.strip() for line in result.split("\n")
                    if line.strip() and not line.strip().startswith(("#", "-", "*", "1.", "2.", "3.", "4.", "5."))
                ]
                # Return original query followed by up to 5 expanded variations
                return [query] + variations[:5]
            except Exception as e:
                print(f"[Query Expansion LLM Fallback: {e}]")

        # Strategy B: Deterministic Domain Expansion (Aligned with coding/data/ manifest)
        expanded = [query]
        q_lower = query.lower()

        # 1. US F-1 OPT & Practical Training (us_dhs_ice_opt_policy_guidance.pdf)
        is_us_work = any(term in q_lower for term in ["opt", "cpt", "ead"]) or (
            any(term in q_lower for term in ["work", "job", "employment", "internship"])
            and not any(c in q_lower for c in ["germany", "france", "spain", "europe"])
        )
        if is_us_work:
            expanded.append(f"{query} post-completion OPT practical training Form I-765 EAD")
            expanded.append(f"{query} direct relationship to major SEVP policy guidance 8 CFR 214.2(f)")

        # 2. US F-1 Status, Form I-20 & Travel (us_uf_f1_graduate_regulations.pdf)
        is_us_visa = any(term in q_lower for term in ["i-20", "sevis", "dso"]) or (
            any(term in q_lower for term in ["visa", "travel", "status"])
            and not any(c in q_lower for c in ["germany", "france", "spain", "europe"])
        )
        if is_us_visa:
            expanded.append(f"{query} Form I-20 travel signature endorsement DSO valid 12 months")
            expanded.append(f"{query} maintaining lawful F-1 status full-time enrollment")

        # 3. Germany Admissions & Study Visa (eu_germany_study_visa_regulations.pdf)
        if any(term in q_lower for term in ["germany", "german", "blocked account", "sperrkonto", "tum"]):
            expanded.append(f"{query} Section 16b AufenthG student visa Aufenthaltserlaubnis")
            expanded.append(f"{query} Sperrkonto blocked bank account health insurance Bürgeramt")

        # 4. France Admissions & Student Visa (eu_france_student_immigration_guide.pdf)
        if any(term in q_lower for term in ["france", "french", "paris", "vls-ts", "ofii", "caf"]):
            expanded.append(f"{query} VLS-TS student visa OFII validation ANEF platform")
            expanded.append(f"{query} CAF housing subsidy APL titre de séjour 964 hours work limit")

        # 5. Spain Admissions & Student Visa (eu_spain_uc3m_immigration_guide.pdf)
        if any(term in q_lower for term in ["spain", "spanish", "madrid", "nie", "tie", "empadronamiento"]):
            expanded.append(f"{query} TIE foreign student identity card NIE number")
            expanded.append(f"{query} estancia por estudios visa empadronamiento town hall")

        # 6. Academic policies, refunds, course drops
        if any(term in q_lower for term in ["drop", "refund", "reduced course load", "rcl", "housing"]):
            expanded.append(f"{query} reduced course load RCL academic drop deadline tuition refund")
            expanded.append(f"{query} residential housing agreement cancellation fee Form SG-402")

        # Deduplicate while preserving order
        return list(dict.fromkeys(expanded))
        

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None,
        resolve_parent: bool = True
    ) -> List[Document]:
        # TODO: Hybrid recall -> Metadata filter -> Cross-Encoder rerank -> Resolve parent
        """
        Executes:
        1. Multi-query retrieval over Ensemble (Chroma + BM25)
        2. Relational metadata filtering
        3. Cross-Encoder joint self-attention re-scoring
        4. Optional Parent-Child hierarchical context resolution
        """
        if not self.ensemble_retriever or not self.vectorstore:
            raise RuntimeError("Index not initialized. Run initialize_retrieval_index() first.")

        # Step 1: Automatic jurisdiction pre-filtering if caller did not supply one
        active_filter = metadata_filter
        if active_filter is None:
            q_lower = query.lower()
            if any(term in q_lower for term in ["germany", "german", "aufenthg", "sperrkonto", "studienkolleg", "tum"]):
                active_filter = {"domain_category": "eu_germany_admissions"}
            elif any(term in q_lower for term in ["france", "french", "paris", "vls-ts", "ofii", "anef", "caf"]):
                active_filter = {"domain_category": "eu_france_admissions"}
            elif any(term in q_lower for term in ["spain", "spanish", "madrid", "tie", "nie", "empadronamiento", "uc3m"]):
                active_filter = {"domain_category": "eu_spain_admissions"}
            elif any(term in q_lower for term in ["opt", "cpt", "ead", "sevis", "8 cfr", "form i-765"]):
                active_filter = {"domain_category": "us_federal_immigration"}

        # Step 2: Multi-query candidate expansion
        query_variation = self.expand_query(query) 
        candidate_pool: List[Document] = []
        seen_texts = set()

        for q_var in query_variation:
            # Dense search with optional relational metadata filter
            if active_filter:
                dense_hits = self.vectorstore.similarity_search(q_var, k=6, filter=active_filter)
            else:
                dense_hits = self.vectorstore.similarity_search(q_var, k=6)

            # Sparse lexical BM25 search
            bm25_hits = self.bm25_retriever.invoke(q_var)
            if active_filter:
                bm25_hits = [
                    d for d in bm25_hits
                    if all(d.metadata.get(k) == v for k, v in active_filter.items())
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

        # Attach scores to candidate metadata
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
                    cached_parent = self.parent_docstore[parent_id]
                    # Create document copy so cached metadata is not mutated
                    parent_doc = Document(
                        page_content=cached_parent.page_content,
                        metadata=dict(cached_parent.metadata)
                    )
                    parent_doc.metadata["cross_encoder_score"] = doc.metadata["cross_encoder_score"]
                    parent_doc.metadata["matched_child_id"] = doc.metadata.get("child_id")
                    parent_doc.metadata["matched_child_content"] = doc.page_content
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
        metadata_filter: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 1.0
    ) -> Dict[str, Any]:
        """
        Executes complete RAG pipeline: retrieves reranked parent contexts,
        constructs cited prompt, evaluates against confidence threshold gate,
        and synthesizes final answer.
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
            sources.append({
                "source": src,
                "page": page,
                "score": round(score, 4),
                "matched_child_id": doc.metadata.get("matched_child_id", "N/A"),
                "parent_id": doc.metadata.get("parent_id", "N/A")
            })
            context_blocks.append(f"[Document {i} | Source: {src}, Page {page}]\n{doc.page_content}")

        context_str = "\n\n".join(context_blocks)

        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are an authoritative international student advisor for StudentGator.\n"
                    "Your knowledge base consists of official regulatory documents from the United States (USCIS/DHS, SEVP, University of Florida), "
                    "Germany (Section 16b AufenthG), France (VLS-TS, OFII), and Spain (TIE, NIE, UC3M).\n\n"
                    "Rules for answering:\n"
                    "1. Rely ONLY on the provided context excerpts. Do not infer, extrapolate, or assume policies not directly supported.\n"
                    "2. Explicitly specify the governing jurisdiction, institution, and official source document.\n"
                    "3. Cite exact statutory terms, form designations (e.g., Form I-20, Form I-765, TIE, NIE, VLS-TS, Sperrkonto), and policy sections whenever mentioned in the excerpts.\n"
                    "4. If the provided excerpts do not contain the complete answer, explicitly state what is missing according to the available documents.\n\n"
                    "Context Excerpts:\n{context}"
                ),
            ),
            ("human", "{question}"),
        ])

        top_score = sources[0]["score"] if sources else -999.0
        if not retrieved_docs or top_score < confidence_threshold:
            answer = (
                "[Grounded Boundary]: The provided official policy documents do not contain specific "
                f"regulations addressing this inquiry (top confidence score {top_score:+.2f} below threshold {confidence_threshold:+.2f}). "
                "Please consult the competent authority directly (e.g., Ausländerbehörde in Germany, USCIS in the US, Préfecture in France) "
                "or contact the StudentGator advisory desk at advising@studentgator.com."
            )
        elif self.llm:
            chain = prompt_template | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context_str, "question": query})
        else:
            # Deterministic fallback response when running locally without active LLM API keys
            top_src = sources[0]["source"] if sources else "None"
            top_page = sources[0]["page"] if sources else "N/A"
            answer = (
                f"[Grounded Synthesis from {top_src} (Page {top_page})]:\n"
                f"{retrieved_docs[0].page_content}"
            )

        return {
            "query": query,
            "query_variations": self.expand_query(query),
            "answer": answer,
            "sources": sources,
            "context_used": context_str,
            "retrieved_docs": retrieved_docs,
            "confidence_threshold": confidence_threshold
        }

    def print_pipeline_diagnostics(
        self,
        rag_result: Dict[str, Any],
        show_full_context: bool = False
    ) -> None:
        """
        Prints detailed, human-readable inspection diagnostics for a RAG pipeline run:
        1. Target query and generated semantic query expansions.
        2. Candidate retrieval, Cross-Encoder re-rank scores, and Parent-Child resolution.
        3. Final grounded synthesis and verified source citations.
        4. Optional complete injected context inspection.
        """
        query = rag_result.get("query", "")
        variations = rag_result.get("query_variations", [])
        retrieved_docs = rag_result.get("retrieved_docs", [])
        sources = rag_result.get("sources", [])
        answer = rag_result.get("answer", "")
        context_used = rag_result.get("context_used", "")
        threshold = rag_result.get("confidence_threshold", 1.0)

        top_score = sources[0]["score"] if sources else -999.0
        is_blocked = (not retrieved_docs) or (top_score < threshold)
        status_banner = (
            f"[REJECTED BY THRESHOLD GATE: Score {top_score:+.2f} < {threshold:+.2f}]"
            if is_blocked else
            f"[ACCEPTED BY THRESHOLD GATE: Score {top_score:+.2f} >= {threshold:+.2f}]"
        )

        divider = "=" * 78
        subdivider = "-" * 78

        print(f"\n{divider}")
        print("                 ADVANCED RAG PIPELINE DIAGNOSTIC INSPECTOR")
        print(f"{divider}")
        print(f"Target Query : \"{query}\"")
        print(f"Gate Status  : {status_banner}")

        # 1. Semantic Expansions
        print(f"\n[1. Semantic Query Expansions ({len(variations)} total)]")
        for idx, var in enumerate(variations, start=1):
            tag = "Original User Query" if idx == 1 else f"Semantic Expansion #{idx-1}"
            print(f"   [{idx}] ({tag}):\n       -> {var}")

        # 2. Re-ranked Documents & Resolution
        print(f"\n[2. Re-Ranked Documents & Parent Context Resolution ({len(retrieved_docs)} returned)]")
        for rank, doc in enumerate(retrieved_docs, start=1):
            meta = doc.metadata
            score = meta.get("cross_encoder_score", 0.0)
            src = meta.get("source", meta.get("file_name", "Unknown"))
            page = meta.get("page", "N/A")
            domain = meta.get("domain_category", "N/A")
            parent_id = meta.get("parent_id", "N/A")
            child_id = meta.get("matched_child_id", "N/A")
            matched_seed = meta.get("matched_child_content", "")

            print(f"\n   --- Rank #{rank} | Cross-Encoder Re-rank Score: {score:+.4f} ---")
            print(f"   Source File : {src} (Page {page}) | Domain: {domain}")
            print(f"   Resolution  : Child '{child_id}' -> Parent '{parent_id}' ({len(doc.page_content)} characters)")
            if matched_seed:
                clean_seed = " ".join(matched_seed.split())
                print(f"   Matched Seed: \"{clean_seed[:130]}...\"")
            clean_parent = " ".join(doc.page_content.split())
            print(f"   Injected Ctx: \"{clean_parent[:180]}...\"")

        # 3. Final Grounded Synthesis & Sources
        print(f"\n[3. Final Grounded Synthesis]")
        print(f"{subdivider}")
        print(answer)
        print(f"{subdivider}")
        print("Verified Sources Cited:")
        for s in sources:
            print(f"   * {s['source']} (Page {s['page']} | Score: {s['score']:+.4f})")

        # 4. Injected Context Inspector (Optional)
        if show_full_context and context_used:
            print(f"\n[4. Full Injected Context Inspection]")
            print(f"{subdivider}")
            print(context_used)
            print(f"{subdivider}")

        print(f"{divider}\n")


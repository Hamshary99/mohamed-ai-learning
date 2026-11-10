The model that generates embeddings is called an encoder.

---

## Pre-Retrieval Optimization Techniques

The key items here are:

* a) How we embed the domain-specific data
* b) How we store it
* c) How we manipulate and transform the query to ensure better retrieval

---

## Query Expansion

* **Purpose:** Maximizes retrieval recall by addressing phrasing mismatches and synonym blind spots in the user's prompt.

Because vector models map semantic coordinates based on overall meaning, slight phrasing discrepancies or out-of-domain vocabulary can place the query vector just outside the neighborhood of the relevant documents.

Query Expansion uses an LLM or Small Language Model (SLM) as a reasoning engine:

1. **Generation:** Generates the prompt into 3 to 5 more prompts with similar intents.
2. **Parallel Retrieval:** All generated intents are embedded and searched against the vector database in parallel.
3. **Candidate Aggregation:** The retrieved document chunks from every query variation are collected and deduplicated, casting a wider net across the vector space to maximize recall.

* **Trade-off:** Increases retrieval recall, but introduces extra latency and token cost.

---

## Multi-Query Retriever & Query Decomposition

Vanilla RAG assumes that every user prompt maps to a **single piece of information**. However, real-world user queries are frequently **compound, multi-part, or comparative** (e.g., *"Which is better, A or B?"*).

A single vector search fails here because there is rarely a single document comparing both:

* Information about **A** is in Document 1.
* Information about **B** is in Document 2.
* A single vector cannot point to two different documents in opposite areas of the database at once.

### How Query Decomposition Works

```
"Which is better, A or B?"
            │
   [LLM Decomposer]
            │
  ┌─────────┴─────────┐
  ▼                   ▼
Sub-Query A         Sub-Query B
("What is A?")      ("What is B?")
  │                   │
  ▼                   ▼
Vector Search       Vector Search
  │                   │
  ▼                   ▼
[Context A]         [Context B]
  └─────────┬─────────┘
            │
            ▼
[Both fed into prompt]
            │
            ▼
Final Comparative Answer
```

### The 4-Step Process

1. **Decomposition:** The LLM inspects the compound query and breaks it into independent, atomic sub-questions.
2. **Parallel Retrieval:** Sub-Query A and Sub-Query B query the vector database in parallel.
3. **Context Fusion:** Context A and Context B are combined into a single unified context window.
4. **Synthesis:** The generator LLM reads both retrieved contexts side-by-side to synthesize a clear comparative answer.

* **Key Implementations:** `MultiQueryRetriever` in LangChain; `SubQuestionQueryEngine` in LlamaIndex.

---

## Parent-Child Retriever (Hierarchical Context Enrichment)

### What Does It Fix? (The Chunk Size Dilemma)

In Vanilla RAG, the exact chunk that is vectorized and searched in the database is the **exact same chunk** that gets injected into the LLM prompt. This creates an impossible compromise:

* **If we use small chunks (e.g. 100–150 tokens):**
  * Vector similarity search works well. Small chunks have concentrated, specific meanings, making it easy to find exact sentences without noise.
  * *The Problem:* The LLM lacks the surrounding context, narrative flow, or qualifying clauses needed to generate a complete answer (e.g. *"It costs \$500"* without explaining *what* costs \$500).
* **If we use large chunks (e.g. 1,000–1,500 tokens):**
  * The LLM has broad context and full paragraphs for a complete answer.
  * *The Problem:* The embedding averages out 20 different ideas into a single vector. This dilutes and blurs out specific details, which degrades search precision.

### The Solution: Decouple Search from Generation

The **Parent-Child Retriever** splits the responsibilities:

* **Child Chunks** are used for **Searching** (high precision).
* **Parent Chunks** are used for **Answering** (rich context).

```
[ Raw Document ]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ Large Parent Chunk (e.g. 800 tokens) — Stored in DocStore   │
│                                                             │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐ │
│   │ Child Chunk 1 │   │ Child Chunk 2 │   │ Child Chunk 3 │ │  (150 tokens each)
│   │ (In VectorDB) │   │ (In VectorDB) │   │ (In VectorDB) │ │   Stored with parent_id
│   └───────────────┘   └───────────────┘   └───────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                ▲
                 User Query ────┘ (Matches Child Chunk 2)
                                │
               Retriever pulls the FULL PARENT CHUNK
                                │
                                ▼
                    Injected into LLM Prompt
```

### The 3-Step Mechanics

1. **Ingestion & Indexing (Small & Big):**
   * The document is first divided into large **Parent Chunks** (e.g. 800 tokens) kept in an underlying document store (in-memory dictionary, Redis, or SQLite).
   * Each parent chunk is subdivided into multiple smaller **Child Chunks** (e.g. 150 tokens).
   * **Only the small child chunks are embedded and stored in the vector database**, linked by metadata to their `parent_id`.
2. **Search (High Precision):**
   * The user query matches against the sharp, focused child chunk vectors in the Vector DB.
3. **Context Enrichment (Auto-Merging):**
   * Once a child chunk is matched, the retriever looks up its metadata `parent_id` and fetches the **full parent chunk** to send to the LLM prompt.
   * If two child chunks from the same parent are matched, the system auto-merges and deduplicates them so the parent chunk is only passed once.

### Trade-Off

* **Pro:** Solves the chunking dilemma completely. You get the high precision of searching tiny chunks without starving the LLM of narrative context.
* **Con:** Requires managing two storage layers (a Vector DB for child vectors + a Document Store for parent text).

* **Key Implementation:** `ParentDocumentRetriever` in LangChain.

---

## Hybrid Search: Semantic (Dense) + Sparse (BM25)

### Why Vector Search Alone Fails on Exact Terms

When we built native RAG, vector search handled broad questions and semantic similarity well. However, dense embeddings struggle when dealing with exact alphanumeric codes, product SKUs, IDs, and proper nouns:

* **Why it happens:** Dense encoders map sentences into continuous concept space. They smooth out exact characters into general topics. To an embedding model, two completely different course codes or policy IDs look identical because they share the same topic.
* **Real-world examples:**
  * **Course codes:** Searching for `"CS-101 prerequisites"`. The vector database sees `"CS-101"` and `"CS-102"` (or `"Intro to Computer Science"` and `"Data Structures"`) in the exact same semantic neighborhood and returns the wrong course syllabus.
  * **Product SKUs / Versions:** Searching for `"iPhone 14 case"`. An iPhone 14 case and an iPhone 15 case have a ~0.98 cosine similarity because they mean almost the same thing conceptually. But physically, an iPhone 14 case does not fit an iPhone 15.
  * **Forms and Policies:** Searching for `"Form I-20 travel endorsement"` for an `"F-1"` international student. Dense search might return general essays on F-1 visa regulations that omit Form I-20 entirely, because it treats them as topical synonyms.

---

### Sparse Lexical Search & BM25 Made Simple

Classical search engines (like Elasticsearch and Lucene) use **Sparse Lexical Search**, specifically **BM25** (Best Matching 25), to guarantee exact keyword matching.

BM25 evolves classical TF-IDF (Term Frequency–Inverse Document Frequency) using 3 simple scoring rules:

1. **Term Frequency (TF) with Saturation (Diminishing Returns):**
   * If a query term appears in a chunk, the chunk gets points.
   * Unlike basic TF-IDF (where 20 mentions gives 20x the score of 1 mention), BM25 has a ceiling. Mentioning a keyword 3 times gives a big boost; mentioning it 20 times gives only a tiny incremental gain. This stops keyword-stuffed spam documents from dominating.
2. **Inverse Document Frequency (IDF) — The Rarity Bonus:**
   * Common words across the entire database (e.g. `"student"`, `"university"`, `"the"`, `"policy"`) get nearly **0 points**.
   * Rare, unique terms (e.g. `"CS-101"`, `"I-20"`, `"SG-402"`) get a **massive point bonus**.
3. **Document Length Normalization:**
   * If a short 40-word chunk mentions `"CS-101"`, it is almost certainly dedicated to that course.
   * If a 5,000-word catalog mentions `"CS-101"` once in an appendix footnote, it is likely irrelevant.
   * BM25 penalizes overly long documents so short, focused chunks win.

**Scoring Walkthrough:**

* Query: `"CS-101 refund"`
  * **Chunk A (Short, 50 words):** *"Students dropping CS-101 before Friday receive a 100% refund."* -> **High BM25 score** (contains the rare token `CS-101`, contains `refund`, short chunk length).
  * **Chunk B (Long, 2,000 words):** General university fee guidelines mentioning `"refund"` 12 times, with `"CS-101"` once in the footer. -> **Lower BM25 score** (penalized by document length, and TF saturation caps the 12 `"refund"` mentions).

---

### Hybrid Search Architecture

Hybrid Search runs both systems in parallel:

* **The Dense Retriever (ChromaDB / Embeddings):** Finds broad concepts, intent, paraphrases, and synonyms.
* **The Sparse Retriever (BM25):** Acts as an exact keyword safeguard for technical codes, names, and error codes.

```
                                  User Query
                          ("Refund policy for SG-402")
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        [ Dense Vector Search ]                [ Sparse Lexical Search ]
         (Embeddings / ChromaDB)                        (BM25)
                    │                                     │
         Captures intent & synonyms            Guarantees exact match
          ("money back", "tuition")             for code "SG-402"
                    │                                     │
                    ▼                                     ▼
          Top-K Ranked Chunks                   Top-K Ranked Chunks
          [Doc A (Rank 1)]                      [Doc B (Rank 1)]
          [Doc C (Rank 2)]                      [Doc A (Rank 2)]
          [Doc D (Rank 3)]                      [Doc E (Rank 3)]
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                     [ Reciprocal Rank Fusion (RRF) ]
                       Combines based on list rank
                                       │
                                       ▼
                           Final Unified Context
                           (Injected into Prompt)
```

---

### The Scoring Challenge: Why We Cannot Just Add Scores

* **Cosine Similarity** produces bounded decimal numbers between `0.0` and `1.0` (e.g. `0.82`).
* **BM25** produces unbounded positive numbers based on term statistics (e.g. `8.4`, `24.6`, `51.2`).
* If we simply add them (`Score = Dense + Sparse`), BM25's score of `24.6` will completely drown out the vector score of `0.82`.

### The Solution: Reciprocal Rank Fusion (RRF)

RRF solves this by ignoring raw numerical scores entirely. Instead, it looks only at where each chunk finished in the ranking order (1st place, 2nd place, 3rd place).

**The Formula 1 / Olympic Analogy:**
When athletes compete in events with different scoring units (like racing times in seconds vs jumping distance in meters), judges do not add the raw times and meters together. They award points based on **finishing rank** (1st, 2nd, 3rd). RRF does the same thing.

**How RRF Works:**
$$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + \text{rank}(d)}$$

* $\text{rank}(d)$: The position of document $d$ in retriever $m$'s list ($1, 2, 3, \dots$).
* $k$: A smoothing constant (standard default is **60**). It prevents top-ranked items from completely eclipsing second and third place.
  * 1st place contributes: $\frac{1}{60 + 1} = \frac{1}{61} \approx 0.0164$
  * 2nd place contributes: $\frac{1}{60 + 2} = \frac{1}{62} \approx 0.0161$
  * 10th place contributes: $\frac{1}{60 + 10} = \frac{1}{70} \approx 0.0143$

**Why RRF Is Effective:**

* **Double-Match Boost:** If Document A finishes Rank 2 in Dense search AND Rank 1 in BM25, its scores sum up:
  $$\text{Score}(A) = \frac{1}{60 + 2} + \frac{1}{60 + 1} = \frac{1}{62} + \frac{1}{61} \approx 0.0325$$
  Documents that satisfy both semantic meaning and exact keyword match jump to the very top.
* **Single-List Protection:** If a document only appears in BM25 (e.g. Rank 1 because of an obscure error code) but missed the dense retriever's top results, it still gets $\frac{1}{61} \approx 0.0164$, ensuring crucial exact matches aren't lost.

---

## Metadata Filtering & Enrichment

### Why It Exists (The Problem With Flat Search)

Naive vector search computes cosine distance against every chunk in the database (e.g. 50,000 documents across 20 university departments).

* When an undergrad asks *"health insurance deadlines"*, a flat search might retrieve varsity sports clearance (Athletics), staff benefits (Payroll), or an expired 2021 memo.
* While semantically related, they are factually wrong for a 2026 undergraduate, wasting context tokens and causing hallucinations.
* **Real-World Bug Example (Cross-Jurisdiction Leak):**
  * Query: *"Can international students work in Germany while studying?"*
  * Without pre-filtering (`filter=None`), flat search pulled `us_uf_f1_graduate_regulations.pdf` (University of Florida, USA) as Rank #2.
  * **Why it happened:** 75% of the query tokens are generic concepts (*"international students"*, *"work"*, *"studying"*), with only 25% being *"Germany"*. The U.S. handbook had 60 pages heavily discussing student work hours, so flat vector search leaked U.S. F-1 rules into a German inquiry.
  * **The Fix:** Relational pre-filtering (`WHERE domain_category == 'eu_germany_admissions'`) draws a hard boundary, blocking U.S., French, and Spanish chunks before vector distance math begins.

```
Naive:        Query ──> [ Vector Search all 50,000 chunks ] ──> Noisy, cross-department results
Pre-Filter:   Query ──> [ Relational Filter (year=2026) ] ──> [ Vector Search 200 chunks ] ──> Clean results
```

---

### How It Works (The 3 Core Mechanics)

1. **Metadata Enrichment (Ingestion):** Attach structured key-value attributes to each chunk:

   ```json
   {
     "text": "Undergraduate immunization proof is due August 1...",
     "metadata": { "department": "admissions", "audience": "undergrad", "year": 2026 }
   }
   ```

2. **Pre-Retrieval Filtering (Production Standard):**
   * The database runs a fast relational filter (`WHERE department == 'admissions' AND year == 2026`) *before* calculating vector similarity.
   * Only the surviving 200 chunks undergo cosine similarity math, slashing latency and blocking outdated context.
3. **Post-Retrieval Filtering (Scoring Thresholds & Rejection Gates):**
   * Vector and cross-encoder models always produce a ranked list, even if all candidates are completely irrelevant.
   * Application logic evaluates candidate scores against a **confidence cutoff** before passing context to the generator LLM.

* **Naive RAG vs. Production Thresholding:**
  * **Naive RAG (Blind Top-K):** Hardcodes `top_k = 3` and blindly passes the top 3 documents into the prompt—even if the highest score is negative.
    * *The Failure Mode:* When asking *"What are the requirements to validate a student visa in France?"*, the top chunk was an English document about **Spain** with a negative score (`-1.4153`). Naive RAG still fed it to the LLM, leading to confident hallucinations about Spanish rules for France.
  * **Production RAG (Scoring Threshold):** Enforces a strict score cutoff gate (e.g. `score >= 0.0` for Cross-Encoder logits, or `cosine_similarity >= 0.70`).
    * *The Guardrail:* If the top candidate score falls below the threshold, the pipeline intercepts execution *before* calling the LLM and returns a deterministic fallback: *"The verified database does not contain policies answering this inquiry. Please contact official advising."*

* **Cross-Encoder Score Interpretation (`ms-marco-MiniLM-L-6-v2` logits):**
  * `> +2.0`: Highly confident, direct factual match. Safe to synthesize.
  * `0.0 to +2.0`: Partially relevant or contextual background. Proceed with caution.
  * `< 0.0`: Irrelevant, semantic false friend, or language mismatch. Blocked by the threshold gate.

* **When to Use What:**
  * **Pre-Filtering:** Static categorical filters (`department`, `year`, `jurisdiction`). Fast relational prune that saves vector compute.
  * **Post-Filtering:** Dynamic runtime thresholds (`score >= 0.0`, confidence gates). Eliminates hallucinations when retrieved data is off-topic.

---

### Why for StudentGator (LangGraph Routing)

In Week 3 Day 3, our **LangGraph Router Node** classifies student intent (e.g. visa inquiry) and injects `{"department": "visas", "year": 2026}` directly into ChromaDB as a pre-filter before similarity search:

```python
results = collection.query(
    query_texts=["health insurance submission deadline"],
    n_results=5,
    where={
        "$and": [
            {"department": {"$eq": "international_admissions"}},
            {"year": {"$eq": 2026}},
        ]
    },
)
```

---

## Bi-Encoder vs. Cross-Encoder (Re-Ranking)

### 1. Bi-Encoder (Independent Encoding — Optimized for Speed)

* **How It Works (My Mental Model):**
  This model runs on two parallel encoding streams: one for the query and the other for the documents. Even with thousands of documents, each document passes through the encoder independently to produce a fixed coordinate vector.

```
Query Stream:     User Query ──────> [ Encoder ] ──────> Vector Q
                                                            │ (Fast Cosine / Dot Product)
Document Stream:  Doc Chunks ──────> [ Encoder ] ──────> Vector D   │
```

* **Why It Is Fast (The Pro):**
  All document vectors are pre-computed offline and stored in ChromaDB. At search time, comparison is pure mathematical vector distance ($O(1)$ or approximate nearest neighbors).

* **The Fatal Flaw (The Con):** Loss of token-to-token interaction.
  * A Bi-Encoder compresses an entire chunk (e.g. 500 characters) into a single coordinate point (e.g. 384 or 1536 floats).
  * Because documents were encoded days ago in isolation, **the encoder has no idea what query will be asked**. It cannot highlight specific words or model the interaction between query tokens and document tokens.
  * **Failure Example (Negation & Nuance):**
    * Query: *"Can international students work off-campus without authorization?"*
    * A Bi-Encoder assigns high similarity to chunks discussing *"international students working on-campus with authorization"* because almost all words match semantically. The critical qualifier *"without authorization"* gets smoothed out in the vector average.

---

### 2. Cross-Encoder (Joint Encoding — Optimized for Accuracy)

* **What "Cross" Actually Means:**
  In a Bi-Encoder, the query and document never interact during encoding. In a Cross-Encoder, the query and document chunk are concatenated into a **single sequence** and fed into a Transformer model together:
  `[CLS] User Query [SEP] Document Chunk [SEP]`
* **Full Cross-Attention:**
  Because both enter the neural network simultaneously, every single word in the query attends to (compares itself directly against) every single word in the document. It immediately catches word order, negation (*"not"*, *"without"*), and fine-grained nuances that standalone vectors miss.
* **No Vector Embeddings Produced:**
  It does not output a vector. Its classification head outputs a single scalar value: a **relevance score between 0.0 and 1.0**.

**The 3-Step Sequence:**

1. **Pairing:** Concatenate the query and candidate chunk: `[CLS, query, SEP, chunk]`.
2. **Joint Cross-Attention:** Feed both into the Transformer simultaneously so every query token attends to every document token.
3. **Direct Scoring:** Output a single similarity score from `0.0` to `1.0`.

```
[Query + Chunk A ("without authorization")] ──> [Cross-Encoder] ──> Score: 0.96 (Relevant)
[Query + Chunk B ("with authorization")]    ──> [Cross-Encoder] ──> Score: 0.14 (Irrelevant)
```

* **The Fatal Flaw:** Extreme computational cost.
  Because the query and document must be processed together, **we cannot pre-compute embeddings beforehand**. If a database has 100,000 chunks, running 100,000 Transformer forward passes per query would take several minutes, making it unusable for primary search.

---

### 3. The Solution: Two-Stage Re-Ranking Pipeline

In production, we combine both models into a two-stage retrieval funnel:

```
User Query ──> [ Stage 1: Bi-Encoder / Hybrid ] ──> Top 50 Candidates ──> [ Stage 2: Cross-Encoder ] ──> Top 3 Chunks ──> LLM Prompt
               (Speed & Recall across 100k chunks)                       (Precision & Nuance Re-ranking)
```

1. **Stage 1 (Bi-Encoder / Hybrid Search) — High Recall:**
   * Rapidly scans 100,000+ pre-indexed chunks in milliseconds.
   * Goal: Pull the top 50 broadly relevant candidate chunks.
2. **Stage 2 (Cross-Encoder Re-Ranker) — High Precision:**
   * Takes **only those 50 candidate chunks** and evaluates each jointly with the user query.
   * Discards semantic false matches (e.g. confusing "with" and "without") and sorts by true relevance.
   * Passes the top 3–5 clean, verified chunks to the LLM prompt.

---

### Comparison: Bi-Encoder vs. Cross-Encoder

| Feature | Bi-Encoder (Embedding Model) | Cross-Encoder (Re-Ranker) |
| :--- | :--- | :--- |
| **Input** | Query and Document encoded separately | Query and Document concatenated as a pair |
| **Output** | Dense vector embeddings ($d$-dimensional) | Single relevance score (`0.0` to `1.0`) |
| **Attention** | Self-attention within query and doc independently | Full cross-attention between query and doc tokens |
| **Pre-Computation** | Yes: Chunks are indexed and stored offline | No: Must be computed dynamically at query time |
| **Speed / Cost** | Extremely fast ($O(1)$ lookup via Vector DB) | Slow (Transformer forward pass per candidate pair) |
| **Pipeline Role** | Stage 1: Broad candidate retrieval (High Recall) | Stage 2: Fine re-ranking of top candidates (High Precision) |

---

### Production Code Reference (Sentence-Transformers Cross-Encoder)

```python
from sentence_transformers import CrossEncoder

# Load a lightweight, pre-trained re-ranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "Can international students work off-campus without authorization?"
candidate_chunks = [
    "F-1 students may engage in on-campus employment up to 20 hours per week with valid status.",
    "Working off-campus without prior USCIS authorization is a severe violation of student status.",
    "Optional Practical Training (OPT) allows off-campus work authorization after graduation.",
]

# Create (query, chunk) pairs for joint scoring
pairs = [[query, chunk] for chunk in candidate_chunks]
scores = reranker.predict(pairs)

# Rank candidates by score in descending order
ranked_results = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
# ranked_results[0] -> Chunk 2 with high relevance score
```

---

## Ensemble Retrieval & Weighted Strategies

### The Problem: Every Single Retriever Has a Blind Spot

When designing a search system, no single algorithm works for every query:

* **Dense Vector Search** understands concepts and broad semantic meaning, but frequently misses exact product numbers, error codes, and acronyms.
* **BM25 Keyword Search** catches exact codes and proper nouns, but fails completely on synonyms.
* **Multi-Query Retrievers** cast a wide net across different phrasings, but can return noisy, loosely related results.

**Ensemble Retrieval** solves this by borrowing the exact same intuition as **Ensemble Learning in classical Machine Learning** (like Random Forests or Gradient Boosting): **combine multiple diverse, independent models to create a single, stronger decision.**

---

### How Combining Retrievers Works in Practice

When a query arrives, the system fans it out across multiple retrieval streams simultaneously:

```
User Query ──┬──> Retriever 1: Dense Vector (ChromaDB) ──> Top 10 Candidates (Semantic) ──┐
             │                                                                         ├──> Weighted RRF ──> Final Merged List
             └──> Retriever 2: Sparse Keyword (BM25)   ──> Top 10 Candidates (Exact)    ┘
```

1. **Parallel Execution:** Dense and Sparse retrievers execute independently and concurrently.
2. **Ranked Outputs:** Each retriever generates its own candidate list according to its internal score.
3. **Fusion (The Hurdle):** Merging disparate candidate lists fairly without letting one algorithm overpower the other.

---

### Incompatible Score Scales & RRF

We cannot simply add raw retrieval scores together:

* **Cosine Similarity** outputs tightly bounded values from `-1.0` to `1.0` (or `0.0` to `1.0`).
* **BM25** outputs unbounded positive numbers (e.g. `8.2`, `17.4`, `32.0`) based on term frequency and document length.
* Adding them directly causes BM25 to completely drown out the vector score every single time.

**Reciprocal Rank Fusion (RRF)** solves this by ignoring raw numbers and ranking candidates purely by **position** (1st, 2nd, 3rd):
$$\text{RRF Score}(d) = \sum_{m \in \text{Retrievers}} \frac{1}{k + \text{rank}_m(d)}$$

* $\text{rank}_m(d)$: The position of document $d$ in retriever $m$'s list.
* $k$: A smoothing constant (standard default is **60**) that balances the scoring curve so multi-retriever consensus items rise to the top.

---

### Weighted Strategies: Balancing Your Retrievers

In practice, we rarely want a 50/50 balance. Depending on the domain and data type, we assign proportional weights:

* **Conversational FAQs & Student Forums:** Users express thoughts in varied, subjective phrasing -> Weight **Dense Vector** higher (e.g. `0.7` Vector / `0.3` BM25).
* **Technical Manuals, Legal Codes, & Course Syllabi:** Exact codes, acronyms, and proper nouns dominate -> Weight **BM25** higher (e.g. `0.6` BM25 / `0.4` Vector).

#### Weighted RRF Formula

Each retriever's RRF contribution is scaled by an assigned importance weight:
$$\text{Weighted Score}(d) = w_{\text{vector}} \cdot \left(\frac{1}{60 + \text{rank}_{\text{vector}}(d)}\right) + w_{\text{bm25}} \cdot \left(\frac{1}{60 + \text{rank}_{\text{bm25}}(d)}\right)$$
If configured with `weights = [0.7, 0.3]`, a document near the top of the vector search gets 70% of the scoring power, while the keyword search provides a 30% boost for exact keyword matches.

---

### Production Code Reference (LangChain EnsembleRetriever)

In our Week 3 pipeline, LangChain's `EnsembleRetriever` runs both retrievers in parallel, applies our weights, and merges the candidate lists via RRF automatically:

```python
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 1. Initialize independent retrievers
vector_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# 2. Combine them into an Ensemble with custom weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 70% Semantic Vector, 30% Lexical BM25
)

# 3. Retrieve fused, ranked documents
final_docs = ensemble_retriever.invoke("What are the requirements for an I-20 travel signature?")
```

* **Week 3 Pipeline Role:** `EnsembleRetriever` serves as the first-stage candidate retrieval engine (High Recall), feeding the top 50 fused candidates into the Cross-Encoder for deep re-ranking (High Precision).

---

## Hands-on Code References for Practice

### 1. Query Expansion (LCEL Pipeline)

Uses an LLM to generate 3 alternative query formulations to maximize vector recall:

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

expansion_prompt = PromptTemplate.from_template(
    "You are an AI assistant helping with document search.\n"
    "Generate 3 different search queries with identical intent for the question below.\n"
    "Output each query on a new line without numbers or dashes.\n\n"
    "Original Question: {question}\n\n"
    "Alternative Queries:"
)

# LCEL Pipeline: Prompt -> LLM -> String Output Parser
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
expansion_chain = expansion_prompt | llm | StrOutputParser()

# Returns a clean list of query variations
raw_output = expansion_chain.invoke({"question": "How do international students renew F-1 visas?"})
expanded_queries = [line.strip() for line in raw_output.strip().split("\n") if line.strip()]
```

---

### 2. Multi-Query Decomposition (Sub-Questions)

Breaks down compound or comparative questions into independent atomic sub-queries:

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

decomposition_prompt = PromptTemplate.from_template(
    "Decompose the following compound question into 2 or 3 independent, atomic sub-questions.\n"
    "Output each sub-question on a new line without numbers or dashes.\n\n"
    "Compound Question: {question}\n\n"
    "Sub-Questions:"
)

decomposition_chain = decomposition_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.0) | StrOutputParser()

compound_query = "What is the difference between CPT and OPT work authorization for international students?"
raw_sub_queries = decomposition_chain.invoke({"question": compound_query})
sub_queries = [query.strip() for query in raw_sub_queries.strip().split("\n") if query.strip()]
# sub_queries -> ['What are the rules and requirements for CPT?', 'What are the rules and requirements for OPT?']
```

---

### 3. Parent-Child Retriever (Hierarchical Indexing)

Stores small child chunks in ChromaDB for high-precision search, but returns full parent chunks to the LLM:

```python
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Define splitters: Parent (rich narrative context) & Child (sharp search precision)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# 2. Storage layers: Chroma for child vectors, InMemoryStore for full parent text
vectorstore = Chroma(collection_name="child_chunks", embedding_function=OpenAIEmbeddings())
docstore = InMemoryByteStore()

# 3. Initialize ParentDocumentRetriever
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    byte_store=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 4. Ingest raw documents (child vectors and parent documents are indexed automatically)
# parent_retriever.add_documents(raw_documents)
# matched_parent_docs = parent_retriever.invoke("health insurance waiver deadline")
```

---

### 4. Metadata Pre-Filtering (ChromaDB)

Prunes the vector search space using relational metadata before running cosine distance math:

```python
import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection(name="studentgator_knowledge_base")

# Ingest chunks with structured metadata
collection.add(
    documents=[
        "All incoming international students must submit immunization proof by August 1.",
        "Varsity athletes must complete medical clearance before preseason tryouts.",
    ],
    metadatas=[
        {"department": "international_admissions", "year": 2026, "audience": "undergraduate"},
        {"department": "athletics", "year": 2026, "audience": "athletes"},
    ],
    ids=["doc_adm_01", "doc_ath_01"],
)

# Pre-retrieval relational filter: only scans the matching department and year
query_results = collection.query(
    query_texts=["immunization deadline"],
    n_results=2,
    where={
        "$and": [
            {"department": {"$eq": "international_admissions"}},
            {"year": {"$eq": 2026}},
        ]
    },
)
```

---

### 5. Pure Python Reciprocal Rank Fusion (RRF)

Merges independent candidate lists using rank position instead of raw incompatible scores:

```python
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    Combines multiple ranked lists of document IDs using Reciprocal Rank Fusion.
    Formula: Score(d) = sum(1 / (k + rank)) across all lists where d appears.
    """
    fused_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank_idx, doc_id in enumerate(ranked_list, start=1):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += 1.0 / (k + rank_idx)

    # Sort documents by descending RRF score
    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

# Example: Fusing Dense candidate IDs and BM25 candidate IDs
dense_candidates = ["chunk_A", "chunk_B", "chunk_C", "chunk_D"]
bm25_candidates = ["chunk_C", "chunk_A", "chunk_E", "chunk_F"]

fused_rankings = reciprocal_rank_fusion([dense_candidates, bm25_candidates], k=60)
# Result: chunk_A and chunk_C appear in both lists and surge to the top
```

---

### 6. Cross-Encoder Re-Ranking (Sentence-Transformers)

Second-stage re-ranking using full cross-attention on `(query, document)` pairs:

```python
from sentence_transformers import CrossEncoder

# 1. Load lightweight, pre-trained re-ranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "Can international students work off-campus without authorization?"
candidates = [
    "F-1 students may engage in on-campus employment up to 20 hours per week with valid status.",
    "Working off-campus without prior USCIS authorization is a severe violation of student status.",
    "Optional Practical Training (OPT) allows off-campus work authorization after graduation.",
]

# 2. Format into (query, chunk) pairs for joint scoring
candidate_pairs = [[query, chunk] for chunk in candidates]
similarity_scores = reranker.predict(candidate_pairs)

# 3. Sort by relevance score in descending order
ranked_results = sorted(zip(candidates, similarity_scores), key=lambda x: x[1], reverse=True)
# ranked_results[0] -> The exact chunk discussing "without authorization"
```

---

### 7. LangChain EnsembleRetriever (Weighted Dense + Sparse)

Combines dense semantic vector search and sparse lexical BM25 search with domain weights:

```python
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. Initialize independent Dense and Sparse retrievers
vector_retriever = Chroma(
    collection_name="faqs",
    embedding_function=OpenAIEmbeddings(),
).as_retriever(search_kwargs={"k": 10})

bm25_retriever = BM25Retriever.from_documents(sample_documents)
bm25_retriever.k = 10

# 2. Combine into an Ensemble with custom domain weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3],  # 70% Semantic Vector, 30% Lexical BM25
)

# 3. Retrieve fused, deduplicated candidate chunks
final_documents = ensemble_retriever.invoke("What are the requirements for an I-20 travel signature?")
```

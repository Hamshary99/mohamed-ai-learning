"""
Keyword Search vs. Semantic Embedding Search Comparison
-------------------------------------------------------
Demonstrates why standard keyword matching fails on real-world questions,
and how sentence embeddings (Sentence-Transformers) solve it by searching
for MEANING rather than exact words.
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util

# ==============================================================================
# 1. The Knowledge Base (10 Paragraphs from StudentGator University Guidelines)
# ==============================================================================
documents = [
    # 0
    "Annual academic fees for undergraduate programs are set at $14,500 per year, payable in two installments.",
    # 1
    "On-campus residential halls and housing facilities are available for all first-year international students.",
    # 2
    "Applicants from non-English speaking countries must submit an IELTS score of at least 6.5 or a TOEFL iBT score above 85.",
    # 3
    "The final application deadline for the Fall semester intake is July 15th, while Spring applications close on November 1st.",
    # 4
    "Comprehensive medical coverage and emergency healthcare plans are mandatory for all registered students.",
    # 5
    "International students must apply for a study visa at their nearest embassy upon receiving their formal acceptance letter.",
    # 6
    "Enrolled students are legally permitted to work part-time up to 20 hours per week during active academic semesters.",
    # 7
    "Merit-based scholarships covering up to 50% of tuition are awarded to students with a cumulative GPA exceeding 3.8.",
    # 8
    "Course enrollment opens two weeks prior to semester start, and students may drop classes without penalty during the first 10 days.",
    # 9
    "The campus learning center and quiet study rooms remain open 24 hours a day with high-speed internet access."
]


# ==============================================================================
# 2. Search Method A: Simple Keyword Search
# ==============================================================================
def keyword_search(user_query: str, docs: list[str], top_k: int = 1) -> list[tuple[int, float, list[str], str]]:
    """
    Counts how many words from the query appear in each document (case-insensitive).
    Returns list of (doc_index, match_score, matched_words, doc_text) sorted by best match.
    """
    # Extract query words (ignoring short punctuation/stop words)
    query_words = set(user_query.lower().replace("?", "").replace(",", "").replace(".", "").split())
    # Remove basic stop words so we focus on meaningful keywords
    stop_words = {"what", "is", "the", "a", "an", "for", "to", "in", "on", "can", "i", "do", "how", "much", "where", "or", "with"}
    meaningful_query_words = query_words - stop_words

    if not meaningful_query_words:
        meaningful_query_words = query_words

    scores = []
    for idx, doc in enumerate(docs):
        doc_words = set(doc.lower().replace(".", "").replace(",", "").split())
        # Find which exact keywords overlapped
        matched_words = sorted(list(meaningful_query_words.intersection(doc_words)))
        scores.append((idx, float(len(matched_words)), matched_words, doc))

    # Sort descending by match count
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


# ==============================================================================
# 3. Search Method B: Semantic Embedding Search (Sentence Transformers)
# ==============================================================================
print("Loading 'all-MiniLM-L6-v2' embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings for all 10 documents (stored in vector space)
print("Computing embeddings for the 10 documents...\n")
document_embeddings = embedding_model.encode(documents, convert_to_tensor=True)


def semantic_search(user_query: str, docs: list[str], top_k: int = 1) -> list[tuple[int, float, str]]:
    """
    Converts user query into an embedding vector, then calculates cosine similarity
    against all document embeddings.
    Returns list of (doc_index, similarity_score, doc_text) sorted by highest similarity.
    """
    # 1. Embed the query
    query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)

    # 2. Compute Cosine Similarities against all documents
    similarity_scores = util.cos_sim(query_embedding, document_embeddings)[0].cpu().numpy()

    # 3. Rank documents
    ranked_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        results.append((int(idx), float(similarity_scores[idx]), docs[idx]))
    return results


# ==============================================================================
# 4. Side-by-Side Comparison on Test Queries
# ==============================================================================
test_queries = [
    # Test 1: Synonyms (User says "lodging" / "dorms", doc says "residential halls" / "housing facilities")
    "Where can I find student lodging or dorms?",
    
    # Test 2: Synonyms (User says "college cost", doc says "academic fees" / "$14,500")
    "How much does college cost?",
    
    # Test 3: Synonyms (User says "earn money" / "job", doc says "work part-time" / "20 hours per week")
    "Can I earn money with a job while studying?",
    
    # Test 4: Exact Keyword (Both query and doc explicitly say "IELTS")
    "What minimum IELTS score is required?"
]

print("=" * 80)
print("SEARCH SHOWDOWN: KEYWORD SEARCH vs. SEMANTIC EMBEDDING SEARCH")
print("=" * 80)

for query_number, query in enumerate(test_queries, 1):
    print(f"\n[Query #{query_number}]: \"{query}\"")
    print("-" * 80)

    # 1. Keyword Search
    kw_idx, kw_score, kw_matches, kw_text = keyword_search(query, documents, top_k=1)[0]
    if kw_score > 0:
        matches_str = ", ".join(f"'{w}'" for w in kw_matches)
        print(f"  [Keyword Search] (Found {int(kw_score)} matching key{'s' if kw_score > 1 else ''}: [{matches_str}])")
        print(f"     Doc #{kw_idx}: \"{kw_text}\"")
        kw_success = True
    else:
        print("  [Keyword Search] (Score: 0 matches)")
        print("     -> FAILED: None of the query words exist in ANY document!")
        kw_success = False

    # 2. Semantic Search
    sem_idx, sem_score, sem_text = semantic_search(query, documents, top_k=1)[0]
    print(f"\n  [Embedding Search] (Cosine Similarity: {sem_score:.3f} / 1.000)")
    print(f"     Doc #{sem_idx}: \"{sem_text}\"")

    # 3. Verdict
    print("\n  >> Verdict:")
    if not kw_success:
        print("     Keyword search FAILED (0 matches), while Embedding Search correctly understood")
        print(f"     the intent and found Doc #{sem_idx} with high semantic similarity!")
    elif kw_idx == sem_idx:
        print("     Both methods succeeded because the user happened to use the exact keyword.")
    else:
        print("     Keyword search matched accidental words, while Embedding Search found the actual meaning.")

print("\n" + "=" * 80)
print("Takeaway for RAG:")
print("Users rarely use the exact same vocabulary as your legal / university documents.")
print("Semantic search understands INTENT and MEANING, making it the essential backbone of RAG.")
print("=" * 80)

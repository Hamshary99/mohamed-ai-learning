The following benchmark demonstrates the qualitative and quantitative differences across the reference StudentGator dataset:

| Test Question                                                                        | Naive RAG Failure Mode                                                                                                                                       | Advanced RAG Behavior &amp; Grounding                                                                                                                                      | Key Differentiator                                                                     |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Q1: "Hello, what can you help me with?"**                                          | Performs unnecessary vector search across policy docs; returns random text chunks about tuition or insurance; charges retrieval latency.                     | Router node classifies intent as `"greeting"`; branches to `direct_node`; returns instant capabilities overview without searching vector store.                        | **Conditional Edge Routing**: Zero retrieval cost on non-knowledge queries.            |
| **Q2: "What is the penalty for canceling Form SG-402 on August 1st?"**               | Embeds query into vector space; retrieves general housing paragraphs or roommate rules; misses specific cancellation deadline fees due to semantic dilution. | BM25 catches exact keyword `"SG-402"`; dense search catches concept of cancellation; Cross-Encoder scores the July 16–August 15 penalty ($1,200) as highest relevance. | **Hybrid Search &amp; Lexical Matching**: Guarantees exact alphanumeric code matching.     |
| **Q3: "How long is my I-20 travel signature valid if I am on post-completion OPT?"** | Retrieves general 12-month travel rule because "OPT" is treated as an obscure 3-letter acronym; produces an incorrect or contradictory response.             | Hybrid search isolates the exact clause specifying 6-month validity for OPT; Cross-Encoder ranks OPT clause above the general 12-month student clause.                 | **Cross-Encoder Re-Ranking**: Resolves entity relationships and contextual exceptions. |
| **Q4: "Can I get a refund on the $350 international student fee?"**                  | Matches generic refund sentences stating "100% refund granted"; incorrectly informs student they can receive a full refund.                                  | Cross-Encoder isolates the explicit exception: *"international student onboarding fees ($350) are non-refundable"*, preventing false promises.                         | **Precision Truncation**: Strips out contradictory general policy context.             |
| **Q5: "What deductible is required for Form SG-INS-W?"**                             | Fails to retrieve insurance waiver document because user asked about "deductible" rather than "health plan".                                                 | Dense vector search bridges semantic distance between "deductible" and "insurance policy limits"; BM25 verifies exact form code `"SG-INS-W"`.                          | **Reciprocal Rank Fusion**: Balances conceptual synonyms with statutory codes.         |

---

## 8\. Checkpoint Defense Checklist: Meeting #3 with Orabi

To pass Checkpoint Meeting #3 with distinction, prepare to walk through your code and live system against these core technical defense questions:

### Question 1: How does your FastAPI microservice prevent the event loop from blocking during LLM streaming?

&gt; **Defense**: The endpoint is declared as an `async def` function that returns a `StreamingResponse` wrapping an asynchronous generator. The generator consumes tokens via LangGraph's `astream_events(..., version="v2")`. Because network I/O to the LLM provider yields execution back to the `asyncio` event loop on every token packet, a single Uvicorn worker can hold hundreds of concurrent streams without blocking other incoming HTTP connections. Furthermore, client polling checks `request.is_disconnected()` to immediately cancel in-flight model generations if the user closes their browser.

### Question 2: Why combine BM25 and ChromaDB using Reciprocal Rank Fusion rather than adding raw similarity scores?

&gt; **Defense**: BM25 and vector embeddings produce fundamentally incompatible numeric scales. BM25 outputs unbounded positive numbers based on term frequency and document length (often ranging from 4.0 to 45.0+), while cosine similarity outputs normalized floats between -1.0 and 1.0\. Directly summing or averaging these values allows the magnitude of BM25 to completely overpower the dense vector score. Reciprocal Rank Fusion evaluates documents based strictly on their ordinal rank positions ($\\frac{1}{60 + \\text{rank}}$), allowing disparate search strategies to contribute fairly to the final candidate pool.

### Question 3: What is the architectural difference between a Worker Node and a Router in LangGraph?

&gt; **Defense**: A Worker Node is a state-mutating function that performs business logic (such as invoking an LLM, searching ChromaDB, or querying an external database) and returns a dictionary of state updates. A Router is a pure decision gate called within a conditional edge. It reads the current state snapshot and returns a string naming the next node to execute. Routers never mutate state, call external models, or execute heavy computation, ensuring that routing decisions remain isolated, testable, and deterministic.

### Question 4: Where do Design Patterns fit into your capstone project?

&gt; **Defense**: The project implements four core software engineering design patterns:
&gt; 
&gt; 1. **Strategy Pattern**: Implemented in `AdvancedRetrieverService` to allow swapping between Naive Vector Search and Weighted Hybrid Search without altering downstream consumer nodes.
&gt; 2. **State Machine Pattern**: Implemented via LangGraph's `StateGraph` to manage cyclic transitions, multi-turn state, and decision branching.
&gt; 3. **Dependency Injection**: Used in FastAPI routes (`Depends(get_graph)`) to supply pre-compiled graph singletons to endpoint handlers.
&gt; 4. **Observer / Pub-Sub Pattern**: Implemented via Server-Sent Events (SSE) and `app.astream_events()`, where token emissions from the model publisher are pushed to the HTTP subscriber stream.
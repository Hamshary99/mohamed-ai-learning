# LangChain

## 1. What is LangChain?
LangChain is an orchestration framework for building applications with Large Language Models.

While a single LLM API call is simple, real-world systems require an entire pipeline:
* Loading external files (PDFs, docs, websites).
* Slicing text into chunks with overlap.
* Generating embeddings and storing them in vector databases.
* Retrieving relevant chunks based on user questions.
* Injecting context and past conversation history into prompts.
* Parsing outputs into clean strings or structured JSON.

LangChain provides standard, pre-built building blocks to connect all these components together.

---

## 2. Why Use LangChain for RAG?
Without LangChain, I would have to write custom file parsers, manual chunking loops, and database connection logic from scratch.

LangChain standardizes RAG into four stages:
1. **Ingestion:** DocumentLoaders load data into standard `Document` objects (`text` + `metadata`).
2. **Chunking:** `RecursiveCharacterTextSplitter` cuts text into chunks while preserving paragraph/sentence structure.
3. **Storage & Search:** VectorStores (like ChromaDB) index embeddings and turn into search functions via `.as_retriever()`.
4. **Generation:** LCEL pipes (`|`) connect the retriever, prompt template, LLM, and output parser into a single callable pipeline.

### Raw Python vs. LangChain
* **File Reading:** Raw Python needs custom PDF/HTML parsers. LangChain uses one-line loaders (`PyPDFLoader`, `WebBaseLoader`).
* **Chunking:** Raw Python needs manual string index slicing. LangChain uses `RecursiveCharacterTextSplitter`.
* **Vector Search:** Raw Python requires manual vector DB client queries. LangChain uses `.as_retriever(search_kwargs={"k": 2})`.
* **Model Switching:** Raw Python requires rewriting SDK calls. LangChain swaps models by changing one import (`ChatOpenAI` -> `ChatAnthropic`).
* **Pipeline Assembly:** Raw Python nests multiple functions. LangChain connects them with the pipe operator: `retriever | prompt | model | parser`.

---

## 3. Core Building Blocks

### 1. Document Loaders
Pulls data from files or APIs into standardized `Document` objects containing `page_content` and `metadata` (source, page number, timestamp).
* Examples: `PyPDFLoader`, `WebBaseLoader`, `TextLoader`, `CSVLoader`.

### 2. Text Splitters
Automates chunking and overlap to preserve context across cuts.
* Default: `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`.
* Tries splitting on `\n\n` first, then `\n`, then sentences `. `, then words.

### 3. VectorStores & Retrievers
* **VectorStore:** Stores chunks and their embeddings (Chroma, Qdrant, Pinecone).
* **Retriever:** Wraps the database into a search interface:
  ```python
  retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
  ```

### 4. Prompt Templates (`ChatPromptTemplate`)
Reusable prompt blueprints with dynamic variables:
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on this context:\n{context}\n\nQuestion: {question}"
)
```

### 5. Output Parsers
Extracts clean data from the model's raw response object:
* `StrOutputParser()`: Returns only the text string.
* `JsonOutputParser()` / Pydantic: Forces the LLM to output valid JSON matching an exact schema.

### 6. Chains (LCEL - LangChain Expression Language)
Pipes components together using `|` where the output of one step becomes the input to the next:
$$\text{Input} \longrightarrow [\text{Retriever}] \longrightarrow [\text{Prompt}] \longrightarrow [\text{LLM}] \longrightarrow [\text{Output Parser}]$$

### 7. Memory & Chat History
Stores previous messages (`[HumanMessage, AIMessage]`) so the model maintains context for follow-up questions in multi-turn chat.

### 8. Tools & Agents
Functions the LLM can decide to execute when it needs external computation (e.g. calculator, database lookup, live web search).

---

## 4. End-to-End RAG Code Reference

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings # or OpenAIEmbeddings / HuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 1. Load document
loader = TextLoader("handbook.txt")
raw_docs = loader.load()

# 2. Chunk text with overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(raw_docs)

# 3. Store in Vector DB and turn into Retriever
vectorstore = Chroma.from_documents(chunks, embedding=FakeEmbeddings(size=384))
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. Prompt, Model, Parser
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Connect the pipeline using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# 6. Run query
# answer = rag_chain.invoke("What are the requirements?")
# print(answer)
```

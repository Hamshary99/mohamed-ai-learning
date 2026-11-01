# Design Patterns Applied to AI Pipelines

## Group 1: Creational Patterns (How Objects Are Created)

### 1. Singleton Pattern
* **Core Software Concept:** Guarantees that only **one instance** of a class exists across the entire application lifecycle and provides a global access point to it.
* **AI Use Case:** Managing shared vector database client connection pools, or caching heavy local embedding models in RAM/GPU once instead of reloading them on every API request.

**Catch / Trade-off:**
* **Testing issues:** A Singleton creates a permanent global state. If Test A runs and modifies the singleton's data, Test B might fail simply because Test A left data behind (during testing we usually need a fresh copy for every call).
* **Thread-safe bottlenecks:** If 50 API requests hit the server at the exact same millisecond trying to write to that one instance, they have to wait in line.

* **Simple Python Example (Embedding Model Cache):**
```python
class EmbeddingModel:
    _instance = None  # Holds the single instance

    def __new__(cls):
        if cls._instance is None:
            print("Loading 2GB model into GPU memory once...")
            cls._instance = super().__new__(cls)
            cls._instance.model_name = "sentence-transformers"
        return cls._instance

# Both variables point to the EXACT same memory location:
model_a = EmbeddingModel()  # Prints: "Loading 2GB model..."
model_b = EmbeddingModel()  # Doesn't print! Reuses model_a
```

---

### 2. Factory Pattern (Factory Method / Abstract Factory)
* **Core Software Concept:** Defines a blueprint for creating an object, centralizing creation logic so caller code doesn't hardcode specific class names. Subclasses can alter the copy they get from the blueprint to suit their needs.
* **AI Use Case:** A `ModelFactory` dynamically instantiates OpenAI, Anthropic, or local HuggingFace/Ollama instances based on configuration, model type, or user tier.

* **Simple Python Example:**
```python
class ModelFactory:
    @staticmethod
    def get_model(user_tier: str):
        if user_tier == "free":
            return "Initialized LLaMA-3-8B (Cheap local)"
        elif user_tier == "premium":
            return "Initialized GPT-4o (Paid cloud)"

# The caller doesn't need to know how models are set up, just asks the factory:
llm = ModelFactory.get_model(user_tier="free")
```

---

### 3. Builder Pattern
* **Core Software Concept:** Constructing a complex object step-by-step instead of passing 10 confusing arguments into one giant `__init__()`.
* **The Mental Model:** Method chaining (like chaining functions with dots in JavaScript).
* **AI Use Case:** Assembling multi-part AI prompts dynamically (system role, RAG context, user input, few-shot examples).

* **Simple Python Example (Assembling an AI Prompt):**
```python
class PromptBuilder:
    def __init__(self):
        self.parts = []

    def add_system_role(self, role: str):
        self.parts.append(f"SYSTEM: {role}")
        return self  # returns self so you can chain methods

    def add_context(self, context: str):
        self.parts.append(f"CONTEXT: {context}")
        return self

    def add_user_query(self, query: str):
        self.parts.append(f"USER: {query}")
        return self

    def build(self) -> str:
        return "\n".join(self.parts)

# Clean, step-by-step chaining:
prompt = (PromptBuilder()
          .add_system_role("You are a math tutor")
          .add_user_query("What is 2 + 2?")
          .build())
```

---

## Group 2: Structural Patterns (How Objects Are Composed)

### 4. Adapter Pattern
* **Core Software Concept:** Translates an incompatible interface of an external library into an interface your application expects.
* **AI Use Case (Unified LLM Wrapper):** OpenAI uses `.chat.completions.create()`, Anthropic uses `.messages.create()`, and Cohere has its own SDK. An adapter wraps these different provider SDKs under a single uniform `.generate()` interface.

* **Simple Python Example:**
```python
# OpenAI uses: client.chat(prompt=...)
# Anthropic uses: client.messages(text=...)

class AnthropicAdapter:
    def __init__(self, anthropic_client):
        self.client = anthropic_client

    def generate(self, prompt: str) -> str:
        # Translates our standard .generate() into Anthropic's method:
        return self.client.messages(text=prompt)
```

---

### 5. Facade Pattern
* **Core Software Concept:** Provides a simplified, high-level interface to a set of complex lower-level subsystems.
* **AI Use Case:** Masking the complex multi-step RAG pipeline (chunking $\rightarrow$ embedding $\rightarrow$ vector search $\rightarrow$ reranking $\rightarrow$ prompt building) behind a single `rag.query(user_input)` call.

* **Simple Python Example:**
```python
class RAGFacade:
    def ask(self, question: str) -> str:
        # Hides 4 complex subsystems from the caller:
        chunks = chunk_documents()
        vectors = search_vector_db(question)
        prompt = build_prompt(question, vectors)
        return call_llm(prompt)

# The user/API only writes one clean line:
bot = RAGFacade()
answer = bot.ask("How do I return a product?")
```

---

## Group 3: Behavioral Patterns (How Objects Communicate)

### 6. Strategy Pattern
* **Core Software Concept:** Provides an encapsulated interface for an interchangeable set of algorithms, making it swappable between needed algorithms with ease.
* **The Mental Model:** Think of Google Maps. It calculates routes using different *strategies*: **Driving**, **Walking**, or **Transit**. The destination is the same, but the algorithm swaps on the fly.
* **AI Use Case:** Easily swapping between different retrieval strategies (fast keyword search vs. semantic vector search vs. hybrid search) based on user query type.

* **Simple Python Example (Swapping Search Engines):**
```python
# Strategy 1: Fast keyword search
class KeywordSearchStrategy:
    def search(self, query): 
        return ["Match on exact word 'refund'"]

# Strategy 2: Smart semantic vector search
class SemanticSearchStrategy:
    def search(self, query): 
        return ["Match on concept 'money back'"]

class SearchEngine:
    def __init__(self, strategy):
        self.strategy = strategy  # Set the current strategy

    def execute(self, query):
        return self.strategy.search(query)

# Easily swap strategies without changing SearchEngine:
engine = SearchEngine(strategy=KeywordSearchStrategy())
engine.strategy = SemanticSearchStrategy()  # Swapped on the fly!
```

---

### 7. Observer Pattern
* **Core Software Concept:** Provides a subscription-based network for several objects listening to one event-producing object. The producer notifies all listeners automatically when a change occurs.
* **The Mental Model:** A **YouTube Channel Subscription**. The YouTuber (Subject) posts a video once, and all subscribers (Observers) get notified automatically.
* **AI Use Case:** Streaming LLM tokens: where the generation loop emits tokens to the frontend UI, observability tracers (like LangSmith), and billing trackers simultaneously.

* **Simple Python Example (Streaming tokens to multiple listeners):**
```python
class LLMGenerator:
    def __init__(self):
        self.subscribers = []

    def subscribe(self, listener_func):
        self.subscribers.append(listener_func)

    def produce_token(self, token: str):
        # Notify EVERY listener that a new token arrived
        for listener in self.subscribers:
            listener(token)

# Listener 1: Prints to user screen
def print_to_screen(token): 
    print(f"UI displays: {token}")

# Listener 2: Word counter for billing
def count_words(token): 
    print(f"Billing: +1 word logged")

llm = LLMGenerator()
llm.subscribe(print_to_screen)
llm.subscribe(count_words)

llm.produce_token("Hello")  # Triggers both functions automatically!
```

---

## Group 4: Architectural / Inversion Pattern

### 8. Dependency Injection (DI)
* **Core Software Concept:** Passing required services into a class or function from the outside (through `__init__`) rather than letting the class hardcode and create them inside itself.
* **The Simple Mental Model (Why it matters for AI):**
  * If a class creates its own `openai.OpenAI()` client inside `__init__`, every automated test you run will hit the real paid OpenAI API, waste money, and fail if the internet is down.
  * With Dependency Injection, you pass the client into `__init__`. In production, you pass the real OpenAI client. In unit tests, you pass a **fake mock client** that returns instant test text for free!
* **AI Use Case:** FastAPI's `Depends()` system: injecting database sessions, configurations, and pre-warmed LLM/RAG clients directly into API endpoints for clean testing and separation.

#### The Bad Way: Hardcoding Dependencies Inside the Class
```python
import openai

class CustomerSupportBot:
    def __init__(self):
        # BAD: The bot builds its own hardcoded dependency inside!
        # Cannot test this without hitting real paid API:
        self.client = openai.OpenAI(api_key="sk-real-secret-key-12345")

    def answer_question(self, user_text: str):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_text}]
        )
        return response.choices[0].message.content
```

#### The Clean Way: Injecting the Dependency from the Outside
```python
class CustomerSupportBot:
    # CLEAN: We "inject" the ai_client into the bot via __init__
    def __init__(self, ai_client):
        self.client = ai_client

    def answer_question(self, user_text: str):
        return self.client.generate(user_text)

# In production: inject real OpenAI client
# In unit testing: inject fake mock client that runs for $0 offline!
```

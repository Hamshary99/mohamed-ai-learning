# Transformers & Attention
####  Why Self-Attention Exists

* **The Problem with RNNs:** They process words one by one. Long-range context fades out due to vanishing gradients, and training cannot run in parallel on GPUs.
* **How Attention Solves It:** Every word in a sentence looks directly at **every other word simultaneously**. Distance between words no longer matters.

## 1. Word Embeddings
* **Embedding** is having a vector of values for each word in the sentence, and embedding is also what turns the words into vectors.
* So **word embedding** is turning each word or token into a vector of numbers that works as **coordinates in space**, where the vector numbers are **learned weights**.
* **Similar words** are placed closer together in vector space, unlike **one-hot encoding** where every word is isolated.
* **Directions in Space:** Moving in a specific direction in space represents a real-world relationship. For example, the step from `father -> mother` is the exact same arrow as `man -> woman`:

![Embedding Vector Space](image.png)

---

## 2. Positional Encoding

### What is "Encoding"?
In machine learning, **encoding** simply means converting real-world information (like a word, an image, or a position in line) into a **vector of numbers** so a computer can do math on it:
* **Word Embedding:** Encodes the *meaning* of the word into numbers.
* **Positional Encoding:** Encodes the *position/order* of the word (1st, 2nd, 3rd) into numbers.

### The Problem
* Transformers look at **all tokens in parallel** to be fast on GPUs, unlike RNNs which read word by word like a conveyor belt.
* Because self-attention processes everything at the exact same time, it treats the sentence like an unordered bag of words. Without word order, `"Dog bit man"` and `"Man bit dog"` look 100% identical to it!

### The Fix
Before layer 1, we add a Positional Vector ($PE$) directly to the word embedding vector:
$$\vec{X} = \text{Embedding} + PE$$
Now every vector carries both its dictionary meaning and where it sits in the sentence.

### Sine & Cosine Frequencies (How the Position Numbers Are Made)
Instead of just numbering words with raw counts like 1, 2, 3, Transformers use sine and cosine waves of different frequencies:
* Think of it like **clock hands** (second hand moves fast, minute hand moves medium, hour hand moves slow).
* **Keeps values bounded between `[-1, 1]`:** Numbers stay balanced and never blow up.
* **Handles any sentence length:** The waves repeat smoothly, so it can handle short or long text easily.
* **Relative distance:** Helps the model calculate how close or far apart two tokens are from each other.

---

## 3. How Attention Updates Embeddings with Context
* Some words have the exact same spelling but carry **multiple different meanings** (for example, a **mole** can be an animal, a chemistry/mathematical constant, a mark on the skin, or a spiteful rat in a gang).
* At first, the raw **embedding** gives `"mole"` the exact same starting vector values every time.
* Then **Attention** steps in to figure out the context based on the previous and next tokens, nudging the vector to move toward whichever `"mole"` we actually need in vector space.
* So **Attention** is moving the values between embeddings, updating **every single word in the sentence at the same time** with proper context. (When predicting the next word, the final function looks at the last token because it now carries all the combined context from previous tokens, way better than a sole word on its own).
* Same goes for a typical word like `"tower"`: if preceded by `"Eiffel"`, it will move to another vector specific for **Eiffel Tower**.

---

## 4. Queries and Keys

* **Query:** The method of trying to figure out the preceding words or tokens is done by a query function, which is a smaller dimensional set of vectors created by multiplying a weight matrix with the embedded vectors. It acts as a word asking a question (like `"creature"` asking: *"Are there any adjectives in front of me?"*).
* **Key:** Works the exact same way by multiplying a second weight matrix with the embedded vectors to get another smaller set of vectors. It acts as the answer to the Query's question (like `"fluffy"` and `"blue"` answering: *"I'm an adjective!"*).

![Query-Key Attention Grid](image-1.png)

* **Matching Keys and Queries:** We multiply each row's Key with each column's Query (dot product) to get the scores in the grid:
  * **High positive numbers (bright white)** mean they match (`"fluffy"` `+93.0` and `"blue"` `+93.4` match `"creature"`, and `"verdant"` `+93.8` matches `"forest"`).
  * **Negative numbers** mean they don't relate (`"verdant"` gets `-59.8` for `"creature"` because its position is at the end of the sentence).
  * **Softmax** is then used to convert these numbers into a `0` to `1` range instead of $-\infty$ to $+\infty$.

---

## 5. Masking (Preventing Future Tokens from Influencing Past Ones)
* During training (for example, on the sentence `"the cat sat on the chair"`), we reward or punish the model for its assumptions as we progress through the sentence, judging what it expects right after each word.
* **The Cheating Problem:** Since the entire sentence `"the cat sat on the chair"` is mapped in the matrix at once, when the model predicts whether word 3 (`"sat"`) comes after word 2 (`"cat"`), word 2 could inspect word 3's value in the matrix and just copy/memorize it without learning *why* word 3 happened (which fails during real generation when future words like `"sat on the chair"` don't exist yet).
* **How Masking Stops It:** Masking blocks every row after the current word so word 2 (`"cat"`) can only look back at word 1 (`"the"`) and itself. Before Softmax, we set those future cells to **$-\infty$** so they cleanly softmax to `0` (since setting them to a raw `0` instead would mess up the Softmax normalization).

---

## 6. Values & How Q, K, V Work Together

### How Q, K, and V Are Created from Each Word
When a word enters Attention as an embedding vector ($x$), we don't use the raw embedding directly. Instead, we multiply $x$ by **3 separate learned weight matrices**:
* **$Q = x \cdot W_Q$ (Query / What I Need):** Asks *"What kind of context or info does this word need from other words?"*
* **$K = x \cdot W_K$ (Key / What I Have):** Answers *"What kind of info does this word offer to other words?"*
* **$V = x \cdot W_V$ (Value / What I Pass Over):** Packages *"If another word finds me relevant, what actual values should I pass along to update its meaning (coordinates)?"*

### Moving the Word to Its New Coordinates Using Values
* **What Values do:** While **Keys and Queries** only calculate the `0` to `1` Softmax weights in the grid (*how much* attention to pay), **Values** are what actually convert the embedding of one word to its new vector coordinates.
* **Making a Weighted Sum for New Coordinates:**
  * Multiplying the Value matrix ($W_V$) with each word's embedding gives a **Value vector** ($\vec{V}_1, \vec{V}_2, \dots$).
  * We pick one word column (like `"creature"`, $\vec{E}_4$) and multiply each row's **Value vector** by its Softmax weight from the grid, **making a weighted sum** (`0.42 * V2 + 0.58 * V3` = $\Delta \vec{E}_4$).
  * Adding that weighted sum ($\Delta \vec{E}_4$) to the original `"creature"` embedding ($\vec{E}_4$) moves it to new vector coordinates ($\vec{E}'_4$) representing a **fluffy blue creature**:

![Values Weighted Sum Updating Creature Coordinates](image-2.png)

* **This Whole Process is One Single Head of Attention:**
  * Everything above (one Query matrix, one Key matrix, and one Value matrix updating the sequence of embeddings) is called a **single head of attention**.
  * Because Queries, Keys, and Values use much smaller vector dimensions than the full embedding, a single head has a relatively small amount of parameters and only tracks **one specific type of relationship** (like adjectives describing a noun), which is what makes us resort to **Multi-Head Attention**.

---

## 7. Multi-Head Attention

* **What it is:** Running multiple attention heads in parallel, where each head has its own $Q, K, V$ matrices.
* **Why we use it:** A single head only tracks one kind of pattern (like adjectives for a noun). Multiple heads let the model look at the sentence from many distinct perspectives at once (grammar, pronouns, tense, adjectives).
* **Combining Heads:** Instead of just summing them up, the outputs from all heads are **concatenated (glued side by side)** and passed through a final weight matrix ($W_O$) to bring them back to the original embedding size.

![Multi-Head Attention](image-3.png)

---

## 8. The Three Transformer Flavors & Cross-Attention

### 1. Encoder (e.g., BERT)
* **Attention:** Bidirectional (no masking).
* **How it works:** Looks at past and future words at the exact same time to understand the full sentence.
* **Best for:** Sentiment analysis, classification, search, and embeddings.

### 2. Decoder (e.g., GPT)
* **Attention:** Causal Masked attention ($-\infty$ on future tokens).
* **How it works:** Predicts the next word one by one without looking ahead.
* **Best for:** Text generation, chatbots, and writing code.

### 3. Encoder-Decoder (e.g., T5, Original Transformer)
* **Attention:** Combines an Encoder, a Masked Decoder, and **Cross-Attention**.
* **Best for:** Translation and summarization (turning Text A into Text B).

### How Cross-Attention Works (English to Arabic Translation)
Inside each Decoder block, there are two attention steps:
1. **Masked Self-Attention:** The Arabic words generated so far attend to previous Arabic words to keep Arabic grammar consistent.
2. **Cross-Attention:**
   * **Queries ($Q$) come from the Decoder (Arabic):** *"I just generated 'القط' (the cat), which English word should I translate next?"*
   * **Keys ($K$) and Values ($V$) come from the Encoder (English):** The English original text holds the answers. The Keys match the query to `"black"`, and the Values pass along the information so the Decoder generates `"الأسود"`.
   * **Analogy:** The Decoder uses its Arabic sentence as a search Query into the English text's Keys, retrieves the relevant English Values, and uses them to output the next Arabic word.


---

## 9. Evolution of Embeddings: Word2Vec, GloVe & Sentence Embeddings

### 1. Word2Vec: "Guess the Blank"
* **How it works:** It works like how phones guess the next word in a sentence (a sliding window game). For example:
  > *"I poured milk into my coffee and stirred it with a [ _____ ]."*
* It learns to cluster words like `spoon`, `straw`, `fork`, or `knife` close to each other in vector space, far away from unrelated words like `plane` or `cop`.
* By repeating this fill-in-the-blank game millions of times, words used in similar contexts get placed near each other, giving us cool vector math like:
  $$\text{King} - \text{Man} + \text{Woman} \approx \text{Queen}$$
* **The Flaw (Static Coordinates):** It gives each word only **one fixed coordinate**. Words with multiple meanings (like `"bank"` in *money bank* vs. *river bank*) are trapped at the exact same point in space.

---

### 2. GloVe: "The Global Spreadsheet Counter"
Instead of guessing words one sentence at a time like Word2Vec:
* **How it works:** **GloVe** (Global Vectors) scans a massive dataset (like all of Wikipedia) and builds a huge co-occurrence table (spreadsheet).
* It counts: *"How many times across the whole internet does word X appear near word Y?"*
* **A Simple Example (Coffee vs. Tea):**
  * How often does `"coffee"` appear near `"cup"`, `"mug"`, or `"caffeine"`? **A lot.**
  * How often does `"tea"` appear near `"cup"`, `"mug"`, or `"caffeine"`? **Also a lot.**
  * Because their neighbor counts look almost identical across Wikipedia, GloVe places `"coffee"` and `"tea"` close together in vector space, even if they rarely appear in the exact same sentence.
* **The Flaw:** Like Word2Vec, it is still **static** (each word gets only one fixed coordinate, with no context awareness for words like `"bank"` or `"mole"`).

---

### 3. Sentence Embeddings: "The Summary Fingerprint"
Used when you want to compare whole sentences and set coordinates for the **entire sentence**, not just single words:
* In older methods like **Bag of Words**, `"The dog bit the man"` and `"The man bit the dog"` would look 100% identical because they contain the exact same words.
* **Sentence Embeddings (SBERT / Transformers):**
  * Reads the **entire sentence as a whole** with word order intact.
  * Turns the whole sentence into a single **"meaning fingerprint"** vector.
  * For example, *"I love my puppy"* and *"Dogs are my favorite animal"* get almost identical vector coordinates, even though they share zero words!
* **Why do we need this?** It powers **Semantic Search** (like Google Search) and **Vector Databases**: when you search a question, it finds answers matching the *meaning*, not just exact matching keywords.

---

### Quick Comparison
* **Word2Vec:** Learns by playing fill-in-the-blank on small sliding text windows. Produces static word coordinates.
* **GloVe:** Learns by counting how often words appear together across the entire corpus. Also produces static word coordinates.
* **Sentence Embeddings (SBERT):** Reads full sentences through a Transformer to build one contextual meaning fingerprint per sentence.

---

## 10. Vector Similarity: Cosine Similarity vs. Dot Product

Once sentences or documents are converted into vector embeddings, search engines and vector databases need a metric to determine which vectors are closest to each other.

### 1. Cosine Similarity (Angle ONLY)
* **How it works:** It looks strictly at the **angle** between two vector arrows and completely ignores how long they are (magnitude invariant).
* **Score range:** $-1.0$ to $+1.0$ ($+1.0$ means pointing in identical directions, $0.0$ means orthogonal/unrelated, $-1.0$ means diametrically opposite).
* **Mathematical Formula:**
  $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|} = \frac{\sum_{i=1}^d A_i B_i}{\sqrt{\sum_{i=1}^d A_i^2} \cdot \sqrt{\sum_{i=1}^d B_i^2}}$$
* **The Upside (Fairness):** Because it divides out vector length, magnitude does not skew results. A short query or 5-word sentence can closely match a detailed paragraph if they point in the same semantic direction.

---

### 2. Dot Product (Angle AND Length)
* **How it works:** It multiplies corresponding vector components directly and sums them:
  $$A \cdot B = \sum_{i=1}^d A_i B_i = \|A\| \cdot \|B\| \cdot \cos(\theta)$$
* **Score range:** $(-\infty, +\infty)$ (unbounded).
* **The Catch:** Because it does not divide out vector lengths, a gigantic arrow (e.g., a vector with large magnitude) can produce an artificially massive dot product score simply because it has bigger raw numbers, even if its directional alignment is slightly off.

---

### 3. The Engineering "Aha!" Moment ($L_2$ Normalization)

Why do production vector databases (Pinecone, ChromaDB, FAISS, Milvus) care so much about this difference?

If you **normalize** all your vector arrows upfront so every single arrow has an exact length of $1$ (called **unit length** or **$L_2$ normalization**):
$$\|A\| = 1 \quad \text{and} \quad \|B\| = 1$$

Then look at the Cosine formula again:
$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{1 \times 1} = A \cdot B = \text{Dot Product}$$

* **The Takeaway:** When embeddings are $L_2$-normalized, **Cosine Similarity and Dot Product produce identical rankings**.
* **Why Vector Databases Do This:**
  * Calculating cosine similarity on raw vectors requires square roots and division on every single search query.
  * Normalizing vectors once during ingestion allows the database to run pure dot products at search time: simple hardware multiplication and addition (SIMD/BLAS), which is orders of magnitude faster.

---

### Comparison Summary

| Feature | Cosine Similarity | Dot Product | Normalized Dot Product ($L_2$) |
| :--- | :--- | :--- | :--- |
| **What it measures** | Direction / Angle only | Direction + Magnitude | Direction only (Identical to Cosine) |
| **Formula** | $\frac{A \cdot B}{\|A\| \cdot \|B\|}$ | $A \cdot B$ | $A \cdot B$ (where $\|A\|=\|B\|=1$) |
| **Score Range** | $[-1.0, +1.0]$ | $(-\infty, +\infty)$ | $[-1.0, +1.0]$ |
| **Length Sensitivity** | Invariant (ignores length) | Sensitive (large vectors score higher) | Invariant (lengths are fixed to 1) |
| **Search Speed** | Slower (requires norm calculation per query) | Fast | **Fastest** (pure matrix multiplication) |

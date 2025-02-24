### **Lecture 1: NLP Fundamentals – Answers**

1. **Define lexical, syntactic, and tokenization ambiguity in NLP. Provide an example for each.**
	- **Lexical Ambiguity**: A single word has multiple meanings. Example: _bat_ (animal vs. sports equipment).
	- **Syntactic Ambiguity**: A sentence has multiple grammatical interpretations. Example: _I saw the man with a telescope_ (Did I see a man using a telescope or a man who had a telescope?).
	- **Tokenization Ambiguity**: Difficulty in determining word boundaries. Example: _New York_ (should it be split as "New" and "York" or kept together as a single entity?).
2. **Explain the difference between stemming and lemmatization. Why might one be preferred over the other?**
	- **Stemming**: Reduces words to their root form by stripping suffixes. Example: _running → run_ (using rules).
	- **Lemmatization**: Converts words to their dictionary form. Example: _better → good_ (using linguistic knowledge).
	- **Preference**: Lemmatization is preferred when preserving meaning is crucial, whereas stemming is faster but may produce non-existent words.
3. **What are the key challenges in tokenization? How do rule-based and statistical approaches address these challenges?**
	- **Challenges**: Handling contractions (_don't → do + not_), compound words, punctuation, and multi-word expressions.
	- **Rule-Based**: Uses hand-crafted rules (e.g., regex).
	- **Statistical Approaches**: Use probability-based models (e.g., Hidden Markov Models) to predict token boundaries.
4. **Given the following words, apply Porter’s stemming algorithm: "relational," "monitoring," "grasses."**
	- **relational → relate**
	- **monitoring → monitor**
	- **grasses → grass**
5. **Explain the formula for tokenization: $T = \{w_1, w_2, ..., w_n\}$. What does each variable represent?**
	- $T$ represents a set of tokens.
	- $w_1, w_2, ..., w_n$ are the individual tokens extracted from a text input.
6. **Describe the process of Part-of-Speech (POS) tagging and its role in NLP.**
	- POS tagging assigns words in a sentence to grammatical categories (e.g., noun, verb, adjective).
	- **Role**: Helps disambiguate meaning and is essential for parsing, NER, and machine translation.
7. **What is Named Entity Recognition (NER), and why is it important? Provide an example.**
	- **NER**: Identifies entities like names, locations, and dates in text.
	- **Importance**: Used in information extraction, chatbots, and search engines.
	- **Example**: _"Barack Obama was born in Hawaii in 1961."_ → {Barack Obama: PERSON, Hawaii: LOCATION, 1961: DATE}
8. **How do neural network models (e.g., BERT, GPT) differ from statistical models (e.g., HMMs, CRFs) in NLP?**
	- **Statistical Models (HMM, CRFs)**: Rely on probability distributions for sequence modeling.
	- **Neural Networks (BERT, GPT)**: Use deep learning to capture contextual dependencies, often outperforming statistical models in complex NLP tasks.
9. **What are parse trees in syntactic analysis? Draw a parse tree for the sentence: "The cat sat on the mat."**
```
(S
   (NP (Det The) (N cat))
   (VP (V sat)
	  (PP (P on)
		 (NP (Det the) (N mat)))))
```
- **Parse Tree**: Represents the grammatical structure of a sentence.
1. **Discuss the importance of semantic representation in NLP. How is the meaning of a word determined?**
	- **Semantic Representation**: Captures word meanings based on their use in context.
	- **Methods**: Word embeddings (Word2Vec, BERT) compute similarity between words.
	- **Formula**: Meaning($w$) $=$ Context($w$) $+$ Knowledge($w$) where **Context($w$)** is derived from word usage and **Knowledge($w$)** incorporates world knowledge.

---
### **Lecture 2: Text Classification – Answers**
1. **Compare and contrast rule-based classification and supervised learning in text classification.**
	- **Rule-Based Classification**: Uses predefined linguistic rules for classification.
		- ✅ **Pros**: High precision in domain-specific applications.
		- ❌ **Cons**: Hard to scale and maintain.
	- **Supervised Learning**: Uses labeled training data with machine learning models.
		- ✅ **Pros**: More adaptable, easier to automate.
		- ❌ **Cons**: Requires large labeled datasets.
2. **Write the formula for Bayes’ Theorem and explain its components in the context of Naïve Bayes classification.**
	- $P(O|E) = \frac{P(E|O) * P(O)}{P(E)}$
		- **$P(O|E)$**: Probability of class $O$ given evidence $E$.
		- **$P(E|O)$**: Probability of observing $E$ given $O$.
		- **$P(O)$**: Prior probability of class $O$.
		- **$P(E)$**: Probability of observing $E$ across all classes.
3. **How does the independence assumption in Naïve Bayes simplify probability calculations? Provide an example.**
	- **Assumption**: Features are **conditionally independent** given the class.
	- **Simplification**: $P(O|E_1, E_2, ..., E_n) = \frac{P(E_1|O) * P(E_2|O) * ... * P(E_n|O) * P(O)}{P(E_1, E_2, ..., E_n)}$
	- **Example**: In spam filtering, we assume words appear independently: $$P(\text{spam} | \text{"Free money now"}) = P(\text{"Free"} | \text{spam}) * P(\text{"money"} | \text{spam}) * P(\text{"now"} | \text{spam}) * P(\text{spam})$$
4. **Describe how Hidden Markov Models (HMMs) are used in Part-of-Speech (POS) tagging.**
	- **HMMs** model word sequences as hidden states (POS tags) generating observed outputs (words).
	- Uses transition probabilities $P(tag_i | tag_{i-1}))$ and emission probabilities $P(word | tag))$.
5. **What are transition and emission probabilities in HMMs? How are they estimated?**
	- **Transition Probability**: Probability of moving from one state to another: $P(tag_i | tag_{i-1})$
	- **Emission Probability**: Probability of a word given a POS tag: $P(word | tag)$
	- **Estimation**: Based on frequency counts in a labeled dataset.
6. **Explain the Viterbi Algorithm and its role in sequence labeling tasks like POS tagging.**
	- **Purpose**: Finds the most probable sequence of hidden states (POS tags) for a given word sequence.
	- **Algorithm Steps**:
		1. Compute probability of each state at each step using transition and emission probabilities.
		2. Keep track of the best previous state.
		3. Backtrace to find the optimal sequence.
7. **What are the limitations of Naïve Bayes classifiers in real-world NLP tasks?**
	- Assumes feature independence, which is often unrealistic.
	- Sensitive to imbalanced datasets.
	- Performs poorly with correlated features.
8. **Why is text classification important in NLP? Give two real-world applications.**
	- **Importance**: Automates sorting and labeling of text data.
	- **Applications**:
		1. **Spam Detection**: Identifies spam emails based on word patterns.
		2. **Sentiment Analysis**: Determines whether a review is positive or negative.
9. **What is the difference between unigram, bigram, and trigram language models in text classification?**
	- **Unigram**: Uses single words, ignores word order.
	- **Bigram**: Considers word pairs (e.g., _"New York"_ instead of _"New" and "York"_ separately).
	- **Trigram**: Uses sequences of three words for better context capture.
10. **Given a sample dataset of emails labeled as spam or non-spam, describe the steps to train a Naïve Bayes classifier.**
    
	1. **Preprocess Data**: Tokenize and normalize text.
	2. **Calculate Probabilities**:
		- Compute $P(spam)$ and $P(\text{non-spam})$.
		- Compute word likelihoods $P(word | spam)$ and $P(word | \text{non-spam})$.
	3. **Apply Bayes’ Theorem**: Compute the probability of an email being spam given its words.
	4. **Classify**: Assign the email to the class with the highest probability.

---
### **Lecture 3: Information Retrieval – Answers**
1. **What is an inverted index, and why is it important in search engines?**
	- An **inverted index** maps terms to document IDs, enabling efficient document retrieval.
	- **Importance**:
		- Reduces search time by allowing direct access to term-related documents.
		- Stores metadata (e.g., term frequency, document frequency) for ranking.
2. **Explain the differences between exact matching, boolean queries, and phrase queries in IR.**
	- **Exact Matching**: Finds documents containing the exact query term.
	- **Boolean Queries**: Uses logical operators (AND, OR, NOT) to refine searches.
	- **Phrase Queries**: Matches an exact sequence of words (e.g., "climate change policy").
3. **Write and explain the TF-IDF formula. How does it weigh terms in documents?**
	-  $TF-IDF(q, d) = \sum \left(\log(1 + tf_{t,d}) \times \log\left(\frac{D}{df_t}\right) \right)$
	- **$tf_{t,d}$**: Term frequency in document dd.
	- **$df_t$**: Number of documents containing term $t$.
	- **$D$**: Total number of documents.
	- **$\tiny{TF-IDF}$ weights rare terms higher** to emphasize their importance in distinguishing documents.
4. **Why is logarithmic scaling used in $\tiny{TF-IDF}$?**
	- Prevents **bias from high-frequency words** (e.g., "the," "and") by dampening large term frequencies.
	- Improves **discriminative power** of rare words.
5. **Describe the BM25 scoring function and its advantages over TF-IDF.**
	- **BM25 Formula**: $BM25(q, d) = \sum \left( \frac{tf_{t,d} (k1 + 1)}{tf_{t,d} + k1 (1 - b + b \times (dl_d / avgdl))} \times \log \left( \frac{D - df_t + 0.5}{df_t + 0.5} \right) \right)$
	- **Advantages Over TF-IDF**:
		- **Term Saturation**: BM25 controls term frequency influence using k1k1.
		- **Document Length Normalization**: Prevents longer documents from being unfairly favored.
		- **Widely used in search engines** due to better ranking performance.
6. **Define precision, recall, and F1-score in the context of IR evaluation.**
	- **Precision**: Fraction of retrieved documents that are relevant. $\text{Precision} = \frac{TP}{TP + FP}$
	- **Recall**: Fraction of relevant documents that are retrieved. $\text{Recall} = \frac{TP}{TP + FN}$
	- **F1-Score**: Harmonic mean of precision and recall. $F1 = \frac{2 \times (\text{Precision} \times \text{Recall})}{\text{Precision} + \text{Recall}}$
7. **Explain how Mean Reciprocal Rank (MRR) is computed and its significance in search ranking.**
	- **Formula**: $MRR = \frac{1}{|Q|} \sum \frac{1}{\text{rank}_i}$
	- **Interpretation**: Measures how early the first relevant result appears in ranked search results.
	- **Significance**:
		- High MRR = **Users find relevant documents quickly**.
		- Common in **question-answering systems** (e.g., retrieving best forum answers).
8. **What is Normalized Discounted Cumulative Gain (nDCG), and how does it differ from MAP?**
	- **$DCG$ Formula**: $DCG = \sum \frac{\text{rel}_i}{\log_2(i + 1)}$ $nDCG = \frac{DCG}{IDCG}$
	- **Differences from MAP**:
		- **nDCG considers graded relevance** (e.g., ranking "excellent" higher than "good").
		- **MAP** only measures binary relevance (relevant vs. not relevant).
9. **Compare the efficiency of BM25 and TF-IDF in document ranking.**
	- **BM25**: More effective because it accounts for **term frequency saturation** and **document length normalization**.
	- **TF-IDF**: Simpler but less effective in modern search systems.
	- **Efficiency**: BM25 is computationally more expensive but provides **better ranking quality**.
10. **How does query expansion improve search results? Provide an example.**
	- **Definition**: Automatically adding related words or synonyms to improve recall.
	- **Example**:
	    - Query: _"car accident statistics"_
	    - Expanded query: _"car crash statistics," "vehicle collision data," "road accident reports"_.
	- **Benefit**: Helps retrieve documents that use different terminology.

---
### **Lecture 4: Word Representation & Neural IR – Answers**
1. **Describe word embeddings and their advantages over one-hot encoding.**
	- **Word Embeddings**: Dense vector representations of words capturing their semantic relationships.
	- **Advantages Over One-Hot Encoding**:
		- One-hot encoding is sparse (large vectors with mostly zeros).
		- Embeddings capture word **similarities** (e.g., _king_ and _queen_ are close in vector space).
		- Enable **transfer learning** (e.g., Word2Vec, GloVe).
2. **How does Byte Pair Encoding (BPE) help in reducing vocabulary size?**
	- **BPE** is a subword tokenization technique that:
		- **Splits rare words** into frequent subword units (e.g., _"unhappiness" → "un" + "happiness"_).
		- **Reduces Out-of-Vocabulary (OOV) issues** by breaking words into reusable subwords.
		- **Creates a compressed vocabulary** suitable for deep learning models.
3. **Explain how Convolutional Neural Networks (CNNs) are used in Information Retrieval (IR).**
	- CNNs extract **n-gram features** from text.
	- **How it works**:
		- Uses **convolutional filters** over word embeddings to capture local dependencies.
		- Applies **ReLU activation** for non-linearity.
		- Uses **pooling layers** to aggregate key features.
	- **Example Use Case**: CNNs improve **query-document similarity modeling** in IR.
4. **What is the key role of Recurrent Neural Networks (RNNs) in sequence modeling?**
	- RNNs process sequences **one step at a time**, maintaining **hidden states** to store past information.
	- **Role in IR**: Models sequential word dependencies in queries and documents.
	- **Limitation**: Struggles with **long-range dependencies** due to vanishing gradients.
5. **Write the state transition equation for an RNN. What do each of the components represent?**
	- $s_i = g(W_s \cdot s_{i-1} + W_x \cdot x_i + b)$
		- **$s_i$**: Hidden state at step ii.
		- **$x_i$**: Input at step ii (word embedding).
		- **$W_s$, $W_x$**: Trainable weight matrices.
		- **$g$**: Activation function (e.g., $⁡\tanh$, ReLU).
		- **$b$**: Bias term.
6. **Describe the self-attention mechanism used in Transformer models.**
	- **Self-Attention** computes relationships between all words in a sequence.
	- Formula: $\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$
	- **Q (Query), K (Key), V (Value)**: Derived from input word embeddings.
	- **Key Feature**: Enables **parallel processing**, unlike RNNs.
7. **Explain the input format of BERT and why it includes special tokens like $[CLS]$ and $[SEP]$.**
	- **Input Format**: $\quad \text{sentence}_1 \quad [SEP] \quad \text{sentence}_2 \quad [SEP]$
	- **$[CLS]$**: Used for classification tasks.
	- **$[SEP]$**: Separates sentences in **next sentence prediction** tasks.
8. **What is query expansion, and how do word embeddings enhance this process?**
	- **Query Expansion**: Adding related terms to improve search results.
	- **Word Embeddings**:
		- Capture **semantic similarity** (e.g., _"car"_ → _"vehicle"_).
		- Allow **context-aware expansion** (e.g., _"bank"_ → _"finance"_ or _"river"_ depending on context).
9. **Compare CNNs, RNNs, and Transformers in their applicability to NLP tasks.**
	- **CNNs**:
		- Good for **local** patterns (n-gram detection).
		- Efficient for **short text classification**.
	- **RNNs**:
		- Handles **sequential** dependencies.
		- Struggles with **long sequences**.
	- **Transformers**:
		- Uses **self-attention** for **global** context.
		- Scales better for **large datasets**.
10. **What is the significance of fine-tuning pre-trained BERT models for IR tasks?**
	- **Pre-trained BERT** captures general language understanding.
	- **Fine-tuning** adapts it to **domain-specific tasks** like document ranking.
	- Example: BERT fine-tuned for **query-document matching** in IR improves ranking accuracy.

---
### **Lecture 5: Dense Retrieval – Answers**

1. **Describe how dense retrieval differs from traditional first-stage ranking methods like BM25.**
	- **BM25**: Uses term frequency and inverse document frequency for ranking.
	- **Dense Retrieval**:
		- Uses **neural embeddings** instead of keyword matching.
		- Documents and queries are represented as **dense vectors** in a shared vector space.
		- Enables **semantic search**, capturing meaning rather than exact terms.
2. **What are query-passage triples, and why are they important in training dense retrieval models?**
	- **Query-Passage Triples**: (Query, Relevant Document, Non-Relevant Document).
	- **Importance**:
		- Helps the model **differentiate relevant from irrelevant** documents.
		- Used in contrastive learning to **maximize the score margin** between relevant and non-relevant passages.
3. **Compare the Margin Ranking Loss and Binary Cross-Entropy (BCE) loss in retrieval models.**
	- **Margin Ranking Loss**: $\text{Loss} = \max(0, s_{\text{non-rel}} - s_{\text{rel}} + 1)$
		- Encourages relevant scores to be **higher than non-relevant** scores.
	- **Binary Cross-Entropy (BCE) Loss**: $\text{Loss} = BCE(s_{\text{rel}} - s_{\text{non-rel}})$
		- Computes **probability scores** rather than ranking differences.
4. **What are the key differences between brute-force search, Inverted File Index (IVF), and Product Quantization (PQ)?**
	- **Brute-Force Search**:
		- Computes all document-query distances.
		- **Accurate but computationally expensive**.
	- **Inverted File Index (IVF)**:
		- Clusters documents into groups to limit search space.
		- **Efficient but may lose recall**.
	- **Product Quantization (PQ)**:
		- Reduces **vector dimensions** for efficient similarity search.
		- **Fast but may sacrifice precision**.
5. **Explain how BERT-DOT computes document relevance using dot-product similarity.**
	- **Encoding**: q$\hat{q} = BERT([CLS]; q_{1..n})_{CLS}, \quad \hat{p} = BERT([CLS]; p_{1..m})_{CLS}$
	- **Scoring**: $s = \hat{q} \cdot \hat{p}$
	- **Interpretation**: Higher scores indicate higher query-document similarity.
6. **What is Knowledge Distillation, and how does it improve retrieval efficiency?**
	- **Knowledge Distillation**: Transfers knowledge from a large **teacher model** to a smaller **student model**.
	- **How it works**:
		- **Teacher Model**: Produces high-quality ranking scores.
		- **Student Model**: Learns to approximate teacher output, reducing computational cost.
7. **Why do dense retrieval models struggle in zero-shot settings?**
	- **Dense Retrieval models require training on domain-specific data**.
	- Zero-shot settings lack **in-domain training data**, leading to **poor generalization**.
	- BM25 **performs better** across domains due to its rule-based nature.
8. **How does the BEIR Benchmark evaluate retrieval models?**
	- **BEIR** (Benchmarking Information Retrieval) tests retrieval models across **diverse datasets**.
	- Measures **zero-shot retrieval performance**.
	- Finds that **BM25 outperforms dense retrieval in some out-of-domain tasks**.
9. **What are the advantages and limitations of graph-based search methods like HNSW?**
	- **Advantages**:
		- **Fast nearest neighbor search** using hierarchical graphs.
		- **Scales well** for large datasets.
	- **Limitations**:
		- **High memory usage** due to graph structure.
		- **Indexing is expensive**, but retrieval is fast.
10. **How does approximate nearest neighbor (ANN) search improve retrieval efficiency?**
	- **ANN** finds the closest document embeddings without computing all distances.
	- **Techniques**:
	    - **HNSW**: Graph-based nearest neighbor lookup.
	    - **PQ**: Vector compression to reduce memory usage.
	- **Tradeoff**: Sacrifices **some accuracy** for **faster retrieval**.

---
### **Lecture 6: Neural Re-Ranking & Efficient Retrieval – Answers**

1. **Why does training loss not always align with IR evaluation metrics like MRR@10?**
	- Training loss (e.g., **cross-entropy loss**) minimizes classification error but does not directly optimize **ranking metrics**.
	- **MRR@10 (Mean Reciprocal Rank @ 10)** evaluates how early the first relevant document appears, which is not explicitly optimized in standard loss functions.
2. **Compare the efficiency-effectiveness tradeoff between BM25 and BERT-based ranking models.**
	- **BM25**:
		- **Efficient** (fast inference, no deep learning required).
		- **Less effective** (cannot capture semantic meaning).
	- **BERT-based Ranking**:
		- **More effective** (understands query-document relationships deeply).
		- **Computationally expensive** (requires **transformer inference per query-document pair**).
3. **How does the sliding window approach help BERT handle long documents?**
	- **Problem**: BERT has a **512-token limit**.
	- **Solution**:
		- Splits long documents into **overlapping windows**.
		- Computes BERT scores for each window.
		- Uses the **highest-scoring passage** as the document’s final score.
4. **Describe the workflow of neural re-ranking and explain how it improves retrieval accuracy.**
	- **Two-Step Process**:
		1. **First-stage ranker** (e.g., BM25) retrieves **top-k** candidates.
		2. **Neural re-ranker** (e.g., BERT) refines rankings by deeply analyzing relevance.
	- **Improvement**:
		- **Captures semantics** beyond keyword matching.
		- **Better ranking quality** than traditional methods.
5. **Explain the PreTTR method and how it reduces query-time computation in IR.**
	- **Precomputed Term Representations (PreTTR)**:
		- **Precomputes BERT embeddings** for document terms during indexing.
		- During retrieval, only **query embeddings** are computed.
	- **Benefit**: Reduces **real-time BERT computations**, improving efficiency.
6. **What is ColBERT, and how does it improve efficiency in document ranking?**
	- **ColBERT (Contextualized Late Interaction)**:
		- Stores **per-token BERT embeddings** for documents.
		- Query interacts with **stored embeddings**, avoiding full BERT computation.
	- **Efficiency Gain**:
		- Faster retrieval since **document embeddings are precomputed**.
7. **Explain the Mono-Duo pattern and how it refines document ranking.**
	- **Mono-Stage**: Scores query-document pairs independently.
	- **Duo-Stage**: Compares **pairs of documents** to refine ranking.
	- **Benefit**: Captures **relative document importance** in ranking decisions.
8. **Write and explain the formula for cosine similarity in IR.**
	- **Formula**: $\text{sim}(d, q) = \cos(\theta) = \frac{d \cdot q}{|d| |q|}$
	- **Explanation**:
		- Measures the **angle** between document (dd) and query (qq) vectors.
		- Values range from **-1 (opposite)** to **1 (identical)**.
		- Commonly used in **vector-based retrieval models**.
9. **How does MatchPyramid use CNNs for IR? What is the key benefit of this approach?**
	- **MatchPyramid**:
		- Converts query-document interactions into **a 2D matrix**.
		- Applies **CNNs** to learn **local matching patterns**.
	- **Benefit**:
		- Captures **rich text interactions** beyond just term overlap.
10. **What are the challenges of integrating large language models (LLMs) into IR?**
	- **High computational cost**: Transformers require **billions of parameters**.
	- **Latency issues**: Real-time retrieval is challenging due to **slow inference**.
	- **Memory requirements**: Storing **dense embeddings** for documents requires large storage.
	- **Adaptation needed**: LLMs are **not optimized for ranking tasks by default**.

---
### **Lecture 7: N-Gram Models & Language Modeling Fundamentals – Answers**

1. **Why do Out-of-Vocabulary (OOV) words pose a problem in traditional NLP models?**
	- **Problem**: Traditional models rely on fixed vocabularies. If a word is unseen during training, the model **cannot assign it a probability**.
	- **Impact**: Causes **zero probability issues** in statistical models.
	- **Solution**: Use **subword tokenization** (BPE, WordPiece) to handle unknown words.
2. **Explain how smoothing techniques help overcome the zero-probability problem in N-Gram models.**
	- **Problem**: If an unseen n-gram appears in a test sample, its probability is **zero**, making **log-likelihood undefined**.
	- **Smoothing Methods**:
		- **Laplace Smoothing**: Adds **1** to all counts: $P(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n) + 1}{C(w_{n-1}) + V}$
		- **Add-k Smoothing**: Generalized form of Laplace smoothing ($+k$ instead of $+1$).
		- **Backoff & Interpolation**: Use **lower-order n-grams** when higher-order ones are unavailable.
3. **Write the bigram language model formula and explain its significance.**
	- **Formula**: $P(w_1, w_2, ..., w_n) \approx \prod_{k=1}^{n} P(w_k | w_{k-1})$
	- **Significance**:
		- Models word **dependencies** in text.
		- Used in **speech recognition, machine translation, and text prediction**.
4. **How does Byte Pair Encoding (BPE) improve tokenization for NLP tasks?**
	- **Concept**: Merges **frequent character pairs** iteratively.
	- **Advantages**:
		- Handles **rare words** better.
		- Reduces **vocabulary size** while preserving frequent words.
	- **Example**:
        ```
        "unhappiness" → "un" + "happiness"
        ```
5. **Explain the concept of perplexity and how it evaluates language models.**
	- **Definition**: Measures a model’s **uncertainty** in predicting text.
	- **Formula**: $\text{Perplexity}(W) = P(W)^{-1/N}$
	- **Lower perplexity** = Better predictive performance.
6. **What is log probability, and why is it used for numerical stability in NLP computations?**
	- **Problem**: Direct probability multiplication leads to **underflow** (very small numbers).
	- **Solution**: Use **logarithms** to convert multiplication into addition: $\log(P_1 \times P_2 \times P_3) = \log P_1 + \log P_2 + \log P_3$
	- **Benefit**: Improves **computational stability** in probability-based models.
7. **How does WordPiece differ from Byte Pair Encoding?**
	- **WordPiece** selects **pairs that maximize sequence likelihood** (not just frequency).
	- Used in **BERT and Transformer models**.
	- More **probabilistic** than BPE’s purely frequency-based approach.
8. **Describe the role of N-Gram models in text generation and give an example.**
	- **Role**: Predicts **next words** based on previous ones.
	- **Example**:
		- Given _"I want to eat"_, an n-gram model might predict:
			- ```"pizza" (0.45), "sushi" (0.30), "cake" (0.25)```
		- Used in **autocomplete and AI-generated text**.
1. **Compare the advantages and disadvantages of N-Gram models and Transformer-based models.**
	- **N-Gram Models**:
		- ✅ Simple, interpretable.
		- ❌ Cannot model **long-term dependencies**.
	- **Transformers**:
		- ✅ Capture **global context** via self-attention.
		- ❌ Computationally expensive.
2. **How does subword tokenization (BPE, WordPiece) improve model performance over word-level tokenization?**
	- **Word-Level Tokenization Issue**: Large vocabulary and OOV words.
	- **Subword Tokenization Benefits**:
		- **Reduces vocabulary size**.
		- **Handles OOV words** by breaking them into known subwords.
		- **Better generalization** for unseen words.

---

### **Lecture 8: Neural Language Models & Transformers – Answers**

1. **Why do recurrent neural networks (RNNs) struggle with long-range dependencies?**
	- **Issue**: RNNs update their hidden state at each time step, causing earlier information to be **gradually overwritten**.
	- **Vanishing Gradient Problem**: Gradients shrink exponentially during backpropagation, making it difficult to learn long-term dependencies.
	- **Solution**: Use **LSTMs and GRUs**, which retain information over longer sequences.
2. **Explain how LSTMs and GRUs mitigate the vanishing gradient problem in RNNs.**
	- **LSTMs (Long Short-Term Memory)**:
		- Use **gates (input, forget, output)** to control information flow.
		- The **cell state** preserves long-term dependencies.
	- **GRUs (Gated Recurrent Units)**:
		- Similar to LSTMs but **simpler**, using only **reset and update gates**.
	- **Effect**: Helps retain long-term dependencies and reduces vanishing gradients.
3. **Write the formula for self-attention in Transformers and explain its components.**
	- **Formula**: $\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$
	- **Components**:
		- $Q$ (Query), $K$ (Key), $V$ (Value) → Derived from input embeddings.
		- **Softmax** normalizes attention weights.
		- **Scaling ($\sqrt{d_k}$)** prevents extreme softmax outputs.
4. **How does masked attention prevent information leakage during language model training?**
	- **Problem**: Standard self-attention allows words to **attend to future tokens**, which is **not allowed in autoregressive models** (like GPT).
	- **Solution**: **Masking future tokens** by setting their attention scores to $-\infty$, ensuring the model **only attends to previous words**.
5. **Describe the advantages of Transformer-based models over RNNs.**
	- **Parallelization**: RNNs process tokens **sequentially**, while Transformers handle **all tokens simultaneously**.
	- **Long-range dependencies**: Self-attention captures **global relationships** in a sentence, unlike RNNs.
	- **Scalability**: More efficient for **large datasets** and **deep architectures**.
6. **How does the training process of Transformer-based language models differ from N-Gram models?**
	- **N-Gram Models**: Estimate word probabilities based on **fixed context windows**.
	- **Transformers**: Use **self-attention to model dependencies across the entire sequence**.
	- **Training Objective**:
		- BERT: **Masked Language Modeling (MLM)** → Predicts missing words.
		- GPT: **Autoregressive Language Modeling** → Predicts next word sequentially.
7. **Compare fixed-window neural language models and recurrent models. When would each be useful?**
	- **Fixed-Window Models**:
		- Process a **fixed number of previous words**.
		- Used in **simple classification tasks** (e.g., spam detection).
	- **Recurrent Models**:
		- Process **entire sequences** and **store memory** over time.
		- Used in **speech recognition, translation, and text generation**.
8. **Explain the role of tokenization in training neural language models. Why are BPE, WordPiece, and SentencePiece commonly used?**
	- **Role**: Converts text into **numeric representations** for neural models.
	- **Why Use BPE, WordPiece, SentencePiece?**
		- **Handle Out-of-Vocabulary (OOV) words**.
		- **Reduce vocabulary size**.
		- **Improve efficiency** of model training.
9. **Why are Transformers more efficient than RNNs for large-scale NLP tasks?**
	- **Parallel Processing**: Unlike RNNs, Transformers **do not depend on previous states**, enabling **GPU acceleration**.
	- **Better Long-Range Dependencies**: Self-attention captures context **without recurrent connections**.
10. **Describe the key differences between encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) Transformer architectures.**
	- **BERT (Encoder-Only)**:
		- Used for **understanding** tasks (e.g., classification, NER).
		- Trained with **Masked Language Modeling (MLM)**.
	- **GPT (Decoder-Only)**:
		- Used for **generation** tasks (e.g., text completion, chatbots).
		- Uses **autoregressive learning (predicts next token)**.
	- **T5 (Encoder-Decoder)**:
		- Used for **sequence-to-sequence tasks** (e.g., translation, summarization).
		- Converts input to an intermediate representation before generating output.

---
### **Lecture 9: Fine-Tuning & Parameter-Efficient Adaptation – Answers**

1. **What are the challenges of full-model fine-tuning for large language models?**
	- **Computational Cost**: Updating billions of parameters requires **high-end GPUs/TPUs**.
	- **Storage Requirements**: Each fine-tuned version takes **significant disk space**.
	- **Catastrophic Forgetting**: Fine-tuning can cause the model to **forget general knowledge**.
	- **Overfitting**: Small datasets may lead to **overfitting** on specific tasks.
2. **Explain how parameter-efficient fine-tuning (adapters, BitFit) reduces computational costs.**
	- **Adapters**: Add small trainable layers **inside frozen transformer layers**, reducing the number of updated parameters.
	- **BitFit (Bias-Only Fine-Tuning)**: Only **bias terms** in transformer layers are updated, cutting down **memory and compute costs**.
3. **Compare whole-model fine-tuning and head tuning. When would each approach be preferable?**
	- **Whole-Model Fine-Tuning**:
		- Fine-tunes **all parameters**.
		- Needed when **domain shift** is significant (e.g., medical or legal NLP).
	- **Head Tuning**:
		- Only updates the **final classifier layer** while keeping the transformer frozen.
		- Used for **fast adaptation** to new classification tasks.
4. **Write the self-attention formula and explain its role in transformers.**
	- **Formula**: $\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V$
	- **Role**: Allows each token to attend to **all other tokens** in a sequence, improving **context understanding**.
5. **How does retrieval-augmented generation (RAG) extend the capabilities of LLMs?**
	- **Problem**: LLMs are limited by **fixed training data**.
	- **RAG Solution**:
		- Retrieves relevant **external documents** in real-time.
		- Improves factual accuracy and prevents **hallucinations**.
6. **Describe the purpose of using a triangular attention mask in autoregressive text generation.**
	- **Ensures words only attend to previous tokens**, preventing **cheating** during training.
	- Used in **GPT models** to maintain **left-to-right generation**.
7. **What are the key advantages of the encoder-decoder transformer architecture?**
	- **Better suited for sequence-to-sequence tasks** (e.g., translation, summarization).
	- **Encodes full context** before generating output, reducing error propagation.
	- Examples: **T5, BART, mT5**.
8. **Why is self-attention computation quadratic in sequence length? How can this be optimized?**
	- **Quadratic Complexity**: $O(n^2 \cdot d)$ due to pairwise **token interactions** in self-attention.
	- **Optimizations**:
		- **Sparse Attention** (Longformer, BigBird).
		- **Memory-efficient attention** (Reformer, Linformer).
9. **What are the main trade-offs between instruction-tuning and fine-tuning for adapting LLMs?**
	- **Fine-Tuning**: More **task-specific**, needs labeled data.
	- **Instruction-Tuning**: Generalizes better across **multiple NLP tasks**.
	- **Trade-off**: Instruction-tuned models are **more flexible**, but **fine-tuned models are more accurate** for specific applications.
10. **Why do large transformer models suffer from instability during training, and how can this be mitigated?**
	- **Causes of Instability**:
		- **Exploding Gradients** → Use **gradient clipping**.
		- **Poor Weight Initialization** → Use **pre-layer normalization**.
		- **Overfitting on Small Datasets** → Use **dropout & data augmentation**.

---
### **Lecture 10: Prompting Strategies & Instruction-Tuned Models – Answers**

1. **Why do large language models require adaptation techniques like fine-tuning and instruction-tuning?**
	- **Pre-trained LLMs** are trained on broad datasets but lack **task-specific optimization**.
	- **Fine-Tuning**: Adjusts **model weights** to specialize in a specific domain.
	- **Instruction-Tuning**: Trains the model to **follow task instructions more effectively** without full re-training.
2. **Explain how in-context learning (ICL) works and why it is highly sensitive to prompt format.**
	- **ICL Concept**: The model generates responses based on **examples provided in the input prompt**.
	- **Sensitivity to Prompt Format**:
		- **Different wordings** can **drastically change** model outputs.
		- **Order of examples** affects performance.
3. **What is Chain-of-Thought (CoT) prompting, and how does it improve LLM reasoning?**
	- **CoT Prompting**: Encourages the model to generate **intermediate reasoning steps** before the final answer.
	- **Example**:
		- **Basic Prompt**: _"What is 14 × 17?"_ → _"238."_
		- **CoT Prompt**: _"Let's break it down. 14 × 10 = 140, 14 × 7 = 98, so the total is 238."_
	- **Benefit**: Improves **multi-step reasoning** in tasks like math and logic.
4. **Describe the concept of self-consistency in prompting. How does it improve answer reliability?**
	- **Self-Consistency**:
		- Generates **multiple independent responses** to the same prompt.
		- Selects the most common answer.
	- **Benefit**: Reduces **random variability** in model outputs.
5. **How does instruction-tuning differ from standard supervised fine-tuning?**
	- **Supervised Fine-Tuning**: Trains a model on a **specific labeled dataset** for a **single task**.
	- **Instruction-Tuning**: Trains a model on **diverse task instructions**, making it **generalizable across many tasks**.
	- **Example**: Instruction-tuned models like **T5-Instruct, FLAN-T5** perform well on **unseen tasks**.
6. **What are the main causes of bias in large language models? How can they be mitigated?**
	- **Causes**:
		- **Majority Label Bias**: Model over-predicts frequent training labels.
		- **Recency Bias**: Model favors **more recent data** in training corpora.
	- **Mitigation Strategies**:
		- **Debiasing prompts** (e.g., framing questions neutrally).
		- **Fine-tuning on balanced datasets**.
		- **Human feedback (RLHF)** to reinforce ethical guidelines.
7. **Why do structured prompts enhance LLM performance compared to free-form queries?**
	- **Structured Prompts**: Provide **clear instructions** for how the model should respond.
	- **Example**:
		- **Free-Form Query**: _"Tell me about climate change."_
		- **Structured Prompt**: _"Summarize climate change in three sentences with scientific references."_
	- **Benefit**: **Reduces ambiguity**, leading to **more precise and useful outputs**.
8. **Describe the impact of prompt ordering and wording on in-context learning results.**
	- **Order of examples matters**:
		- Early examples influence model predictions **more than later ones**.
	- **Wording sensitivity**:
		- Changing **question phrasing** can lead to **different interpretations**.
		- Example: _"Explain photosynthesis like I'm 5."_ vs. _"Give a scientific definition of photosynthesis."_
9. **What are the key differences between fine-tuning, in-context learning, and instruction-tuning?**
	- **Fine-Tuning**: Changes **model weights** for a **specific dataset**.
	- **In-Context Learning (ICL)**: Uses **examples in the prompt** but **does not change model weights**.
	- **Instruction-Tuning**: Trains a model to **follow task instructions** without needing explicit examples.
10. **How does reinforcement learning from human feedback (RLHF) align LLMs with user intent?**
	- **RLHF Process**:
		1. Human annotators **rank** model responses.
		2. A **reward model** is trained to predict human preferences.
		3. LLM is fine-tuned using **reinforcement learning** to improve responses.
	- **Benefits**:
		- Reduces **harmful outputs**.
		- Makes models **more user-friendly and aligned with human values**.

---
### **Lecture 11: RLHF, Instruction-Tuning, and Long-Context Handling – Answers**

1. **Why is there a mismatch between pre-training and user intent in large language models (LLMs)? How can this be addressed?**
	- **Mismatch**:
		- LLMs are trained on **broad, general datasets** but are often required to perform **specific tasks** for users.
		- Pre-training does not **optimize models for human-aligned behavior**.
	- **Solutions**:
		- **Fine-tuning** on task-specific data.
		- **Instruction-Tuning** to teach models how to respond to various tasks.
		- **Reinforcement Learning from Human Feedback (RLHF)** to align responses with user preferences.
2. **Explain the limitations of instruction-tuning and how RLHF helps overcome them.**
	- **Limitations of Instruction-Tuning**:
		- Requires **large labeled datasets**.
		- Prone to **memorization rather than true generalization**.
		- Can lead to **hallucinations** (incorrect but confident outputs).
	- **How RLHF Helps**:
		- RLHF trains a **reward model** using **human feedback** to adjust model behavior dynamically.
		- Allows models to **refine responses** without needing explicit labels.
3. **What are the main types of bias in language models? How can they be mitigated?**
	- **Types of Bias**:
		- **Majority Label Bias**: Over-reliance on frequent patterns in training data.
		- **Recency Bias**: Preference for more recent information in model outputs.
	- **Mitigation Strategies**:
		- Diversify training datasets.
		- Balance labels to **avoid over-representation** of dominant classes.
		- Apply **bias-regularized training** to reduce skewed responses.
4. **Describe the role of sparse attention mechanisms in handling long contexts. Provide examples of models that use these techniques.**
	- **Sparse Attention** reduces computational complexity by **only attending to a subset of tokens**.
	- **Models Using Sparse Attention**:
		- **Longformer**: Uses **sliding window attention** to extend context length.
		- **BigBird**: Combines **global, local, and random attention**.
		- **Reformer**: Uses **hash-based locality-sensitive attention**.
5. **What is Reinforcement Learning from Human Feedback (RLHF)? Describe its process and the key formula used in reward modeling.**
	- **RLHF Concept**: Uses **human preferences** to refine LLM behavior.
	- **Process**:
		1. Collect human **preference rankings** for multiple model outputs.
		2. Train a **reward model** to predict which outputs are preferable.
		3. Fine-tune the LLM using **reinforcement learning** (PPO algorithm).
	- **Key Formula**: $\hat{\theta} = \arg\max_{\theta} \mathbb{E}_{s \sim p_\theta} R(s; \text{prompt})$
6. **Compare instruction-tuning with fine-tuning in terms of generalization and computational efficiency.**
	- **Fine-Tuning**:
		- Updates model weights for **specific tasks**.
		- Can lead to **overfitting on small datasets**.
	- **Instruction-Tuning**:
		- Teaches the model to follow instructions **across multiple tasks**.
		- More **generalizable**, requiring fewer updates for new tasks.
7. **Explain the benefits and trade-offs of retrieval-augmented generation (RAG). How does it improve LLM performance?**
	- **Benefits**:
		- Reduces **hallucination risk** by retrieving **real-world facts**.
		- Enhances performance on **knowledge-based tasks**.
	- **Trade-Offs**:
		- **Slower** than standard generation models.
		- Requires **efficient retrieval mechanisms** (e.g., FAISS, dense vector search).
8. **What is regularization in RLHF, and why is it necessary? Provide the formula used to prevent reward hacking.**
	- **Regularization Prevents**:
		- **Reward hacking**, where models exploit weaknesses in the reward function.
		- **Deviations from pre-trained distributions**, maintaining fluency.
	- **Formula**: $\hat{R}(s; p) = R(s; p) - \beta \log \frac{p_{RL}(s)}{p_{PT}(s)}$
		- Penalizes large shifts from the **pre-trained model distribution**.
9. **Describe how neural retrieval models, such as ColBERT, enhance the effectiveness of large language models.**
	- **ColBERT (Contextualized Late Interaction)**:
		- Stores **per-token BERT embeddings** for documents.
		- Allows **efficient retrieval without full query-document BERT encoding**.
	- **Enhancements**:
		- Improves **search quality** in retrieval-augmented models.
		- Reduces **computational cost** compared to full BERT ranking.
10. **How does multi-step prompting, such as Chain-of-Thought (CoT), improve reasoning in LLMs? Provide an example.**
- **CoT Prompting**: Guides models to generate **intermediate reasoning steps**.
- **Example**:
- **Basic Prompt**: _"What is 17 × 23?"_ → _"391."_
- **CoT Prompt**: _"17 × 20 = 340, 17 × 3 = 51, so 340 + 51 = 391."_
- **Benefit**: Helps LLMs **solve complex reasoning problems**.

---

### **Lecture 12: Distributed Training, Quantization, and Efficient LLM Architectures – Answers**

1. **What are the major computational challenges in training large language models? List and describe three optimization strategies.**
	- **Challenges**:
		- **Memory Constraints**: LLMs require **hundreds of GBs** of VRAM.
		- **Slow Training Speed**: Billions of parameters lead to **weeks of training**.
		- **Scalability Issues**: Synchronizing updates across **multiple GPUs/TPUs** is complex.
	- **Optimization Strategies**:
		- **Distributed Training**: Uses **data, pipeline, or tensor parallelism** to distribute workloads.
		- **Quantization**: Converts **high-precision parameters** into **lower-bit formats** to save memory.
		- **Knowledge Distillation**: Trains **smaller models** to mimic larger ones for efficiency.
2. **Explain the concept of memory bottlenecks in LLM training and inference. How can activation checkpointing help mitigate this issue?**
	- **Memory Bottlenecks**:
		- Storing **activations** (intermediate values in neural networks) requires **huge memory**.
	- **Activation Checkpointing**:
		- Instead of storing **all activations**, recomputes them **only when needed**.
		- Saves **GPU memory** at the cost of **extra computation**.
3. **What are the key differences between data parallelism, pipeline parallelism, and tensor parallelism in distributed training?**
	- **Data Parallelism**:
		- Each GPU gets a **full copy of the model**, but processes **different batches of data**.
		- Requires **synchronization of gradients** across GPUs.
	- **Pipeline Parallelism**:
		- Splits the **model layers across multiple GPUs**.
		- Reduces **communication overhead** but introduces **pipeline bubbles** (idle GPUs).
	- **Tensor Parallelism**:
		- Splits **individual matrix operations** across GPUs.
		- Best suited for **massively large models** (e.g., GPT-4).
4. **Define quantization in the context of deep learning. Compare post-training quantization (PTQ) with quantization-aware training (QAT).**
	- **Quantization**: Converts **high-precision floating-point numbers** (e.g., FP32) into **lower-bit representations** (e.g., INT8).
	- **PTQ vs. QAT**:
		- **Post-Training Quantization (PTQ)**: Applies quantization **after** training.
			- ✅ Faster, but may lose accuracy.
		- **Quantization-Aware Training (QAT)**: Introduces quantization **during** training.
			- ✅ More accurate, but **computationally expensive**.
5. **What are the different types of model pruning techniques? How do they contribute to efficiency in LLMs?**
- **Magnitude Pruning**: Removes weights with **smallest absolute values**.
	- **Structured Pruning**: Eliminates **entire neurons or layers**.
	- **Unstructured Pruning**: Creates **sparse connections** by randomly removing weights.
	- **Benefits**:
		- **Reduces computational cost** while keeping accuracy intact.
	- Enables **smaller model deployment** for edge computing.
1. **Describe the process of knowledge distillation and explain why it is effective for model compression.**
	- **Process**:
	1. Train a large **teacher model**.
	2. Use its **soft predictions** as labels for a **smaller student model**.
		1. The student model learns to **replicate teacher performance** with fewer parameters.
	- **Effectiveness**:
		- Retains **high accuracy** while reducing **model size** and **inference cost**.
2. **Write and explain the formula for computing FLOPs in transformers. Why is this calculation important?**
	- **Formula**: $C \approx 6ND$ where:
		- $N$ = number of parameters.
		- $D$ = dataset size (tokens).
		- $C$ = total FLOPs required.
	- **Importance**:
		- Helps estimate **computational cost** of training/inference.
		- Guides **hardware optimizations** for efficiency.
3. **How does matrix multiplication affect the computational complexity of transformer models? Provide formulas for forward and backward passes.**
	- **Forward Pass FLOPs**: $2mn$
		- $m$ = number of rows.
		- $n$ = number of columns.
	- **Backward Pass FLOPs**: $4mn$
		- **Twice as costly** due to gradient calculations.
4. **What is mixed precision training, and how does it optimize the balance between speed and accuracy in LLM training?**
	- **Concept**: Uses **lower precision (e.g., FP16, INT8)** for some operations while keeping **high precision (e.g., FP32)** for others.
	- **Benefits**:
		- **Speeds up training** with minimal accuracy loss.
		- **Reduces memory usage**, enabling **larger batch sizes**.
5. **What are the future directions in LLM optimization? Discuss advancements in energy-efficient architectures, sparse computation, and hardware optimizations.**
- **Energy-Efficient Architectures**:
	- Use **low-power accelerators** like **TPUs**.
	- Develop **green AI initiatives** to reduce training footprint.
- **Sparse Computation**:
	- Use **mixture of experts (MoE)** to activate **only parts of the model** at a time.
- **Hardware Optimizations**:
	- Custom **ASIC chips** optimized for transformer computations (e.g., **Tesla Dojo, Google TPU**).
# Classical IR & Feature-Based Methods

| **Technique**      | **Short Description**                                                                                                           | **Pros**                                                                                        | **Cons**                                                                                                       | **Best Usage Scenario**                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **BM25**           | A bag-of-words ranking function using term frequency (TF), inverse document frequency (IDF), and document length normalization. | - Very fast, proven, widely used <br>- Easy to implement and tune <br>- Interpretable scoring   | - Not semantic: purely lexical <br>- Fails on paraphrase or synonyms <br>- Doesn’t leverage contextual meaning | - Good “first-stage” retriever <br>- Resource-constrained search <br>- When interpretability is key |
| **TF-IDF**         | A simpler weighting than BM25; higher emphasis on term frequency and inverse document frequency for ranking.                    | - Intuitive, widely known <br>- Low computation overhead <br>- Good baseline for text retrieval | - Similar limitations as BM25 (purely lexical) <br>- No semantic generalization                                | - Academic exercises or small-scale retrieval <br>- Quick baseline in retrieval experiments         |
| **Inverted Index** | Core data structure for IR systems, storing posting lists of (term, doc ID, frequency).                                         | - Enables very fast lookups for query terms <br>- Scales well to large document sets            | - Large memory footprint for huge corpora <br>- Lexical matching only, no built-in “semantic” understanding    | - Foundational approach for search engines <br>- High-speed term-based retrieval                    |

---

# Neural IR Methods

| **Technique**            | **Short Description**                                                                 | **Pros**                                                                                      | **Cons**                                                                      | **Best Usage Scenario**                                                                                      |
|--------------------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **Dense Retrieval**      | Learns dense vector representations (e.g. BERT/Transformer embeddings) for queries and documents, then uses approximate nearest neighbor search. | - Captures semantic similarity<br>- Can handle synonyms/paraphrases<br>- Good first-stage performance in many tasks | - Requires GPU resources for training<br>- ANN indexing can be complex<br>- May struggle with exact term matching (rare keywords) | - Replacing or complementing BM25 for semantic search<br>- Large-scale retrieval tasks with rich textual data |
| **Neural Re-Ranking**    | Two-stage pipeline: (1) BM25 or dense retrieval for candidate docs, (2) BERT-based re-ranker to refine the top-k results.               | - Typically more accurate than single-stage<br>- Focuses expensive neural inference only on small set of docs | - Requires an initial retriever (BM25/dense)<br>- Still computationally expensive if the top-k is large | - Precision-critical applications (e.g. legal/medical search)<br>- Where compute budget allows a deep re-rank step |
| **Mono- vs. Duo-Stage** | Mono-stage: a single retrieval approach (e.g. pure dense or pure BM25).<br>Duo-stage: a lightweight first stage + neural re-ranking.       | - Mono-stage: simpler, faster, fully end-to-end<br>- Duo-stage: typically higher accuracy and flexible combination | - Mono-stage: can be less accurate<br>- Duo-stage: more complex and expensive to run | - Mono-stage: real-time or resource-limited systems <br>- Duo-stage: critical search tasks needing top precision |
| **HNSW (Approx. NN)**   | Hierarchical Navigable Small World graphs for approximate nearest neighbor search, used to speed up dense retrieval.                     | - Logarithmic-ish search scaling<br>- Handles very large collections efficiently<br>- Easy to tweak parameters for speed vs. accuracy | - Index construction can be memory-intensive<br>- Approximate, so possible retrieval misses | - Large-scale dense retrieval scenarios <br>- Use in combination with BERT/Transformer embeddings |

---

# Language Modeling / Neural Architectures

| **Technique**             | **Short Description**                                                                                          | **Pros**                                                                            | **Cons**                                                                              | **Best Usage Scenario**                                                                                         |
|---------------------------|----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **n-Gram LM**            | Statistical LM that uses fixed-order Markov assumption (e.g. trigram).                                          | - Simple, fast, interpretable <br>- Works fine for short contexts, smaller datasets  | - Poor handling of long-range dependencies <br>- Zero probabilities for OOV or unseen n-grams (requires smoothing) | - Resource-limited tasks <br>- Pedagogical or baseline comparisons                                              |
| **RNN LM**               | Uses recurrent connections (LSTM/GRU) to model variable-length sequences.                                       | - Better at capturing sequential context than n-grams <br>- Less memory usage than large Transformers | - Suffers from vanishing/exploding gradients <br>- Slower to train, not as good with very long contexts          | - Cases with moderate sequence lengths <br>- Real-time streaming or lower-compute environments                  |
| **Transformer LM**       | Self-attention-based approach (e.g. GPT, BERT) to process entire sequences in parallel.                         | - Excellent at handling long-range dependencies <br>- Highly parallelizable at inference time | - Quadratic memory/compute complexity with sequence length <br>- Large model sizes can be expensive             | - Most modern, large-scale LMs <br>- High performance on generative or classification tasks                     |
| **Byte-Pair Encoding**   | Subword tokenization method merging the most frequent character pairs into new tokens.                          | - Reduces OOV issues <br>- Efficient vocabulary size <br>- Widely used in modern LLMs | - Might over-segment or under-segment in rare cases <br>- Some complexity for languages with complex morphology | - Virtually all neural text models, especially large-scale LLMs dealing with diverse text                        |
| **Contextual Embeddings** (e.g., BERT, ELMo) | Word embeddings that adapt to surrounding context so same word can have different vectors in different contexts.  | - Captures polysemy <br>- Greatly improved performance across NLP tasks               | - Typically large models <br>- Hard to interpret                                                              | - Any NLP pipeline needing accurate representation of word meaning in context                                    |

---

# Advanced LLM & Adaptation Techniques

| **Technique**                | **Short Description**                                                                                                     | **Pros**                                                                                                        | **Cons**                                                                                                                      | **Best Usage Scenario**                                                                                                                             |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **Instruction Tuning**       | Fine-tuning an LLM on a dataset of (instruction, response) pairs to make it better at following user instructions.        | - Improves model’s ability to handle diverse tasks <br>- Reduces prompt engineering complexity                                                        | - Requires curated instruction-response data <br>- Overfitting risk if the set of instructions is not diverse                                                   | - Chatbots, multi-domain assistants, or any system where you want user-friendly “instruction -> response” behavior                                     |
| **RLHF (Reinforcement Learning from Human Feedback)** | Policy is updated based on a reward model that reflects human preference (e.g., ranking outputs for better alignment). | - Can greatly improve helpfulness, reduce toxicity<br>- Allows direct “human preference” alignment beyond pure text data                                  | - Collecting high-quality human feedback is expensive <br>- Complex training pipeline (reward model + RL updates)                                              | - Aligning an LLM to domain or ethical constraints <br>- High-stakes applications where human oversight is crucial (customer service, medical)         |
| **Retrieval-Augmented Generation (RAG)** | Combines an LLM with an external retriever so the model can “look up” relevant facts during generation.           | - Reduces hallucination <br>- Easier to update knowledge by refreshing the retriever’s index                                                           | - Requires well-maintained retrieval index <br>- Extra complexity in the pipeline                                                                            | - QA systems for domain knowledge <br>- Dynamic knowledge tasks (where facts change often)                                                                |


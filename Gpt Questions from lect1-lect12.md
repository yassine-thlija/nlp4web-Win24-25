- For the answers visit [[Gpt Questions with solutions]]
### **Lecture 1: NLP Fundamentals**

1. **Define lexical, syntactic, and tokenization ambiguity in NLP. Provide an example for each.**
2. **Explain the difference between stemming and lemmatization. Why might one be preferred over the other?**
3. **What are the key challenges in tokenization? How do rule-based and statistical approaches address these challenges?**
4. **Given the following words, apply Porter’s stemming algorithm: "relational," "monitoring," "grasses."**
5. **Explain the formula for tokenization: $T = {w_1, w_2, ..., w_n}$. What does each variable represent?**
6. **Describe the process of Part-of-Speech (POS) tagging and its role in NLP.**
7. **What is Named Entity Recognition (NER), and why is it important? Provide an example.**
8. **How do neural network models (e.g., BERT, GPT) differ from statistical models (e.g., HMMs, CRFs) in NLP?**
9. **What are parse trees in syntactic analysis? Draw a parse tree for the sentence: "The cat sat on the mat."**
10. **Discuss the importance of semantic representation in NLP. How is the meaning of a word determined?**

---

### **Lecture 2: Text Classification**

1. **Compare and contrast rule-based classification and supervised learning in text classification.**
2. **Write the formula for Bayes’ Theorem and explain its components in the context of Naïve Bayes classification.**
3. **How does the independence assumption in Naïve Bayes simplify probability calculations? Provide an example.**
4. **Describe how Hidden Markov Models (HMMs) are used in Part-of-Speech (POS) tagging.**
5. **What are transition and emission probabilities in HMMs? How are they estimated?**
6. **Explain the Viterbi Algorithm and its role in sequence labeling tasks like POS tagging.**
7. **What are the limitations of Naïve Bayes classifiers in real-world NLP tasks?**
8. **Why is text classification important in NLP? Give two real-world applications.**
9. **What is the difference between unigram, bigram, and trigram language models in text classification?**
10. **Given a sample dataset of emails labeled as spam or non-spam, describe the steps to train a Naïve Bayes classifier.**

---

### **Lecture 3: Information Retrieval (IR)**

1. **What is an inverted index, and why is it important in search engines?**
2. **Explain the differences between exact matching, boolean queries, and phrase queries in IR.**
3. **Write and explain the TF-IDF formula. How does it weigh terms in documents?**
4. **Why is logarithmic scaling used in TF-IDF?**
5. **Describe the BM25 scoring function and its advantages over TF-IDF.**
6. **Define precision, recall, and F1-score in the context of IR evaluation.**
7. **Explain how Mean Reciprocal Rank (MRR) is computed and its significance in search ranking.**
8. **What is Normalized Discounted Cumulative Gain (nDCG), and how does it differ from MAP?**
9. **Compare the efficiency of BM25 and TF-IDF in document ranking.**
10. **How does query expansion improve search results? Provide an example.**

---

### **Lecture 4: Word Representation & Neural IR**

1. **Describe word embeddings and their advantages over one-hot encoding.**
2. **How does Byte Pair Encoding (BPE) help in reducing vocabulary size?**
3. **Explain how Convolutional Neural Networks (CNNs) are used in Information Retrieval (IR).**
4. **What is the key role of Recurrent Neural Networks (RNNs) in sequence modeling?**
5. **Write the state transition equation for an RNN. What do each of the components represent?**
6. **Describe the self-attention mechanism used in Transformer models.**
7. **Explain the input format of BERT and why it includes special tokens like [CLS] and [SEP].**
8. **What is query expansion, and how do word embeddings enhance this process?**
9. **Compare CNNs, RNNs, and Transformers in their applicability to NLP tasks.**
10. **What is the significance of fine-tuning pre-trained BERT models for IR tasks?**

---

### **Lecture 5: Dense Retrieval**

1. **Describe how dense retrieval differs from traditional first-stage ranking methods like BM25.**
2. **What are query-passage triples, and why are they important in training dense retrieval models?**
3. **Compare the Margin Ranking Loss and Binary Cross-Entropy (BCE) loss in retrieval models.**
4. **What are the key differences between brute-force search, Inverted File Index (IVF), and Product Quantization (PQ)?**
5. **Explain how BERT-DOT computes document relevance using dot-product similarity.**
6. **What is Knowledge Distillation, and how does it improve retrieval efficiency?**
7. **Why do dense retrieval models struggle in zero-shot settings?**
8. **How does the BEIR Benchmark evaluate retrieval models?**
9. **What are the advantages and limitations of graph-based search methods like HNSW?**
10. **How does approximate nearest neighbor (ANN) search improve retrieval efficiency?**

---
### **Lecture 6: Neural Re-Ranking & Efficient Retrieval**

1. **Why does training loss not always align with IR evaluation metrics like MRR@10?**
2. **Compare the efficiency-effectiveness tradeoff between BM25 and BERT-based ranking models.**
3. **How does the sliding window approach help BERT handle long documents?**
4. **Describe the workflow of neural re-ranking and explain how it improves retrieval accuracy.**
5. **Explain the PreTTR method and how it reduces query-time computation in IR.**
6. **What is ColBERT, and how does it improve efficiency in document ranking?**
7. **Explain the Mono-Duo pattern and how it refines document ranking.**
8. **Write and explain the formula for cosine similarity in IR.**
9. **How does MatchPyramid use CNNs for IR? What is the key benefit of this approach?**
10. **What are the challenges of integrating large language models (LLMs) into IR?**

---

### **Lecture 7: N-Gram Models & Language Modeling Fundamentals**

1. **Why do Out-of-Vocabulary (OOV) words pose a problem in traditional NLP models?**
2. **Explain how smoothing techniques help overcome the zero-probability problem in N-Gram models.**
3. **Write the bigram language model formula and explain its significance.**
4. **How does Byte Pair Encoding (BPE) improve tokenization for NLP tasks?**
5. **Explain the concept of perplexity and how it evaluates language models.**
6. **What is log probability, and why is it used for numerical stability in NLP computations?**
7. **How does WordPiece differ from Byte Pair Encoding?**
8. **Describe the role of N-Gram models in text generation and give an example.**
9. **Compare the advantages and disadvantages of N-Gram models and Transformer-based models.**
10. **How does subword tokenization (BPE, WordPiece) improve model performance over word-level tokenization?**

---

### **Lecture 8: Neural Language Models & Transformers**

1. **Why do recurrent neural networks (RNNs) struggle with long-range dependencies?**
2. **Explain how LSTMs and GRUs mitigate the vanishing gradient problem in RNNs.**
3. **Write the formula for self-attention in Transformers and explain its components.**
4. **How does masked attention prevent information leakage during language model training?**
5. **Describe the advantages of Transformer-based models over RNNs.**
6. **How does the training process of Transformer-based language models differ from N-Gram models?**
7. **Compare fixed-window neural language models and recurrent models. When would each be useful?**
8. **Explain the role of tokenization in training neural language models. Why are BPE, WordPiece, and SentencePiece commonly used?**
9. **Why are Transformers more efficient than RNNs for large-scale NLP tasks?**
10. **Describe the key differences between encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) Transformer architectures.**

---

### **Lecture 9: Fine-Tuning & Parameter-Efficient Adaptation**

1. **What are the challenges of full-model fine-tuning for large language models?**
2. **Explain how parameter-efficient fine-tuning (adapters, BitFit) reduces computational costs.**
3. **Compare whole-model fine-tuning and head tuning. When would each approach be preferable?**
4. **Write the self-attention formula and explain its role in transformers.**
5. **How does retrieval-augmented generation (RAG) extend the capabilities of LLMs?**
6. **Describe the purpose of using a triangular attention mask in autoregressive text generation.**
7. **What are the key advantages of the encoder-decoder transformer architecture?**
8. **Why is self-attention computation quadratic in sequence length? How can this be optimized?**
9. **What are the main trade-offs between instruction-tuning and fine-tuning for adapting LLMs?**
10. **Why do large transformer models suffer from instability during training, and how can this be mitigated?**

---

### **Lecture 10: Prompting Strategies & Instruction-Tuned Models**

1. **Why do large language models require adaptation techniques like fine-tuning and instruction-tuning?**
2. **Explain how in-context learning (ICL) works and why it is highly sensitive to prompt format.**
3. **What is Chain-of-Thought (CoT) prompting, and how does it improve LLM reasoning?**
4. **Describe the concept of self-consistency in prompting. How does it improve answer reliability?**
5. **How does instruction-tuning differ from standard supervised fine-tuning?**
6. **What are the main causes of bias in large language models? How can they be mitigated?**
7. **Why do structured prompts enhance LLM performance compared to free-form queries?**
8. **Describe the impact of prompt ordering and wording on in-context learning results.**
9. **What are the key differences between fine-tuning, in-context learning, and instruction-tuning?**
10. **How does reinforcement learning from human feedback (RLHF) align LLMs with user intent?**

---

### **Lecture 11: RLHF, Instruction-Tuning, and Long-Context Handling**

1. **Why is there a mismatch between pre-training and user intent in large language models (LLMs)? How can this be addressed?**
2. **Explain the limitations of instruction-tuning and how RLHF helps overcome them.**
3. **What are the main types of bias in language models? How can they be mitigated?**
4. **Describe the role of sparse attention mechanisms in handling long contexts. Provide examples of models that use these techniques.**
5. **What is Reinforcement Learning from Human Feedback (RLHF)? Describe its process and the key formula used in reward modeling.**
6. **Compare instruction-tuning with fine-tuning in terms of generalization and computational efficiency.**
7. **Explain the benefits and trade-offs of retrieval-augmented generation (RAG). How does it improve LLM performance?**
8. **What is regularization in RLHF, and why is it necessary? Provide the formula used to prevent reward hacking.**
9. **Describe how neural retrieval models, such as ColBERT, enhance the effectiveness of large language models.**
10. **How does multi-step prompting, such as Chain-of-Thought (CoT), improve reasoning in LLMs? Provide an example.**

---

### **Lecture 12: Distributed Training, Quantization, and Efficient LLM Architectures**

1. **What are the major computational challenges in training large language models? List and describe three optimization strategies.**
2. **Explain the concept of memory bottlenecks in LLM training and inference. How can activation checkpointing help mitigate this issue?**
3. **What are the key differences between data parallelism, pipeline parallelism, and tensor parallelism in distributed training?**
4. **Define quantization in the context of deep learning. Compare post-training quantization (PTQ) with quantization-aware training (QAT).**
5. **What are the different types of model pruning techniques? How do they contribute to efficiency in LLMs?**
6. **Describe the process of knowledge distillation and explain why it is effective for model compression.**
7. **Write and explain the formula for computing FLOPs in transformers. Why is this calculation important?**
8. **How does matrix multiplication affect the computational complexity of transformer models? Provide formulas for forward and backward passes.**
9. **What is mixed precision training, and how does it optimize the balance between speed and accuracy in LLM training?**
10. **What are the future directions in LLM optimization? Discuss advancements in energy-efficient architectures, sparse computation, and hardware optimizations.**

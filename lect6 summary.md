## Problems
### 1. Mismatch Between Training Loss and IR Evaluation Metrics
- **Issue**: Training loss is not directly comparable to evaluation metrics like **MRR@10**.
- **Impact**: Loss decreases quickly initially but does not directly indicate ranking effectiveness.
### 2. Efficiency-Effectiveness Tradeoff
- **Issue**: More complex models (e.g., BERT) improve ranking quality but significantly increase computational cost.
- **Impact**: High inference time and infrastructure costs.
### 3. Long Document Handling in BERT
- **Issue**: BERT has a maximum input token length of 512.
- **Solution**: Use a sliding window approach and take the max-scored passage.
### 4. Query Latency in Re-Ranking Models
- **Issue**: BERT-based re-ranking requires multiple evaluations, causing slow response times.
- **Solution**: Precompute document embeddings (PreTTR, ColBERT).
## Solutions
### 1. Neural Re-Ranking
- **Concept**: Enhances ranking after initial retrieval.
- **Workflow**:
	- First-stage ranker (e.g., BM25) retrieves top-k documents.
	- Neural re-ranker refines ranking based on deeper analysis.
- **Approaches**:
	- **MatchPyramid** (uses 2D-CNNs for word interaction modeling).
	- **BERT-based Re-Ranking** (uses transformer-based encoding).
### 2. PreTTR (Precomputed Term Representations)
- **Concept**: Precompute early BERT layers for documents.
- **Benefit**: Reduces query-time computations.
- **Formula**: 
	- BERT: 
		- $r=BERT([CLS];q_{1...n};[SEP];p_{1...m})_{CLS}$ (Concatenation)
	- Scoring
	- $s=r*W$ (Starts uninitialized)
- Symbols:
	- $q_{1..n}$ Query Tokens
	- $p_{1..m}$ Passage Tokens
	- $BERT$ pre-trained BERT model
	- $[CLS]$ and $[SEP]$: special tokens
	- $x_{CLS}$ pool the CLS vector
	- $W$ linear layer (from 768 dims to 1)
	- $s$ output score
### 3. ColBERT (Contextualized Late Interaction)
- **Concept**: Precompute and store per-token BERT embeddings for documents.
- **Benefit**: Faster query-time ranking by reducing redundant computations.
- **Formula**:
	- Encoding:
		- $\hat{q} = BERT([CLS]; q_{1..n})_{CLS}$
		- $\hat{p} = BERT([CLS]; p_{1..m})_{CLS}$ (can be done at indexing time)
	- Aggregation:
		- $s = \sum\limits_{i=1..m} \max\limits_{t=1..m}\hat{q}_i\cdot \hat{p}_t$ (Quite efficient at query time)
- Symbols:
	- $q_{1..n}$ Query Tokens
	- $p_{1..m}$ Passage Tokens
	- $BERT$ pre-trained BERT model
	- $[CLS]$ special tokens
	- $s$ output score
### 4. Mono-Duo Pattern
- **Concept**: Multi-stage re-ranking for improved relevance.
- **Stages**:
	- **Mono**: Score query-document pairs (e.g., Top-1000 ranking).
	- **Duo**: Compare document pairs to refine ranking (e.g., Top-50 ranking).
## Formulas
### 1. Cosine Similarity
- $sim(d,q)=\cos⁡(\theta)=\huge{\frac{d\cdot q}{∣d∣∣q∣}}$
- Symbols:
	- $\theta$ Angle between two vectors
	- $q$ vector of query
	- $d$ vector of document
	- $d\cdot q=\sum\limits_{i=1..n}d_i*q_i$ dot product
### 2. BERT-Based Scoring Function
- $r=BERT([CLS];q_{1...n};[SEP];p_{1...m})_{CLS}$
- $s=r*W$
### 3. MatchPyramid Feature Extraction
- **Key Idea**: Transform word similarity into structured representations via CNN layers.
- **Architecture**:
	- 2D Convolutional Layers extract local interaction patterns.
	- Dynamic Pooling adapts to varying input lengths.
	- Fully Connected Layer for final scoring.
## Key Ideas
### 1. Word Representation in Neural IR
- Pre-2019: Word2Vec, GloVe (pre-trained word embeddings).
- Post-2019: BERT (transformer-based contextual embeddings).
### 2. Impact of BERTCAT Re-Ranking
- **Effectiveness**: Significantly improves ranking performance (e.g., MRR@10 increase from .194 to .385 on MSMARCO).
- **Efficiency Tradeoff**: Requires repeated inference for every document-query pair.
### 3. Large Language Models (LLMs) in IR
- **IR for LLM**: Retrieval-Augmented Generation (RAG) enhances generation models.
- **LLM for IR**: Used for query expansion, reformulation, and knowledge graph search.
- **Challenges**: Context-awareness, generalization, and efficiency.
## Summary
- **Neural Re-Ranking**: Enhances document ranking via deep learning models.
- **BERT-Based Methods**: Improve ranking accuracy but require efficiency optimizations.
- **Efficiency Gains**: PreTTR and ColBERT precompute representations to reduce latency.
- **Future Directions**: LLM integration, domain adaptation, and efficiency optimizations.
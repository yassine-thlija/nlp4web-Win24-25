## Dense Retrieval
### Overview
- Dense retrieval replaces traditional first-stage ranking methods like BM25.
- Uses neural encoders and nearest neighbor search.
- Can be used standalone or as part of a larger retrieval pipeline.
### Dense Retrieval Lifecycle
1. **Training**: Models trained using query-passage triples.
2. **Indexing**: Encoding and storing document vectors.
3. **Searching**: Query vectors matched with nearest neighbor document vectors.
### Training Dense Retrieval Models
- Models trained with **triples** (query, relevant doc, non-relevant doc).
- Goal: Maximize the margin between relevant and non-relevant documents.
- Loss functions:
	- **Margin Ranking Loss**:
		- $loss$ = $max(0, s_{nonrel} - s_{rel} + 1)$
	- **Binary Cross-Entropy (RankNet)**:
		- $loss$ = $BCE(s_{rel} - s_{nonrel})$
## Nearest Neighbor Search
### Brute-Force Search
- Computes all document-query distances.
- **Extremely accurate** but computationally expensive.
### Indexing Techniques
- **Flat Index**: No additional processing, best accuracy but slow.
- **Inverted File Index (IVF)**:
	- Clusters dataset into groups.
	- Query is matched to closest cluster, reducing search space.
- **Product Quantization (PQ)**:
	- Converts high-dimensional vectors into lower-dimensional representations.
	- Uses k-means clustering to encode data efficiently.
- **Graph-Based Search (HNSW)**:
	- Uses hierarchical graph structure for efficient nearest neighbor lookup.
	- Highly scalable for large datasets.
## BERT-DOT Model
- Encodes queries and passages into fixed-length vectors.
- Relevance computed using **dot-product similarity**:
	- Encodings:
		- $\hat{q} = BERT([CLS]; q_{1..n})_{CLS}$
		- $\hat{p} = BERT([CLS]; p_{1..m})_{CLS}$
	- Matching:
		- $s = \hat{q}\cdot \hat{p}$
	- Symbols:
		- $q_{q..n}$ are the query tokens
		- $p_{a..m}$ are the passage tokens
		- $BERT$ pre trained BERT model
		- $[CLS]$ special tokens
		- $s$ output score
- Supports **approximate nearest neighbor search** for efficiency.
## Knowledge Distillation in IR
### Why Distillation?
- Large teacher models (e.g., BERT) are **slow** but **accurate**.
- Distillation trains a smaller **student model** to approximate the teacher.
- Reduces computational cost while maintaining effectiveness.
### Distillation Techniques
1. **KL-Divergence**(Kullback-Leibler):
	- Compares probability distributions between teacher and student outputs.
	- Formula:
		- $D_{KL}(P || Q) =\sum_\limits{x\in X} P(x)\log(\frac{P(x)}{Q(x)})$
2. **Margin-MSE Loss**:
	- Optimizes margin between relevant and non-relevant document scores.
	- Works well for **dense retrieval** models.
## Zero-Shot Dense Retrieval
### BEIR Benchmark
- Evaluates generalization of retrieval models across different datasets.
- **Findings:**
	- Dense retrieval models struggle in zero-shot settings.
	- BM25 remains **more stable** across domains.
	- Adaptation techniques improve performance.
## Summary
- **Dense retrieval** improves first-stage ranking using neural networks.
- **Nearest neighbor search** enables efficient retrieval with large-scale indexes.
- **BERT-DOT** provides a simple, effective neural retrieval model.
- **Knowledge distillation** allows smaller, faster models to retain high accuracy.
- **Zero-shot generalization** remains a challenge for dense retrieval models.
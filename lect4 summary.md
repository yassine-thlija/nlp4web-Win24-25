## Word Representation in Information Retrieval
### Key Concepts
- **Word Embeddings**:
	- Dense vector representation of words.
	- Typically 100-300 dimensions.
	- Words with similar meanings are close in vector space.
	- Examples: Word2Vec, GloVe, FastText.
- **Byte Pair Encoding (BPE)**:
	- Subword tokenization technique.
	- Reduces vocabulary size and handles out-of-vocabulary words.
	- Merges frequently occurring symbol pairs iteratively.
## Neural Methods in IR
### Convolutional Neural Networks (CNNs)
- Used for **n-gram representation learning**.
- Apply filters over word sequences to capture local dependencies.
- Example PyTorch implementation:
```python
conv = nn.Sequential(
	nn.ConstantPad1d((0,2 - 1), 0),
	nn.Conv1d(kernel_size=2, in_channels=300, out_channels=200),
	nn.ReLU()
)
embeddings_t = embeddings.transpose(1, 2)
n_grams = conv(embeddings_t).transpose(1, 2)
```
### Recurrent Neural Networks (RNNs)
- Used for **sequence modeling** in IR.
- Maintains memory of previous words to capture long-term dependencies.
- Example equation for an RNN state:
	- $s_i = g(W_s * s_{i-1} + W_x * x_i + b)$
	- where:
		- $s_i$ = hidden state at step $i$.
		- $x_i$ = input at step $i$.
		- $Ws$,$Wx$,$bW_s$, $W_x$, $b$ = trainable parameters.
		- $g$ = activation function (e.g., $\tanh$ or $ReLU$).
### Encoder-Decoder Architecture
- Used for **machine translation, summarization, and question answering**.
- Encoder processes input sequence into a context vector.
- Decoder generates output sequence based on the context.
## Transformer-Based Models
### Self-Attention Mechanism
- Allows each word to focus on all\large{\sum_\limits{t\in T_d \cap T_q}} other words in the sequence.
- Computational complexity: **O(n²)**.
- Essential for deep contextualization.
### BERT (Bidirectional Encoder Representations from Transformers)
- **Key Features**:
    - Uses self-attention to create contextualized word embeddings.
    - Pre-trained on large corpora using **Masked Language Modeling (MLM)**.
    - Input format includes:
        - **\[CLS\]**: Classification token.
        - **\[SEP\]**: Sentence separator.
        - **Positional embeddings**.
- **Example Use Case**: Fine-tuned for text ranking in IR.
## Query Expansion with Word Embeddings
- Expands queries by adding semantically similar words.
- Example:
```
query = "sample query"
expanded_terms = {"example": 0.94, "sampling": 0.87, "inquire": 0.76}
```
- Improves retrieval performance by increasing recall.
## Summary
- **Word embeddings** provide dense representations for words.
- **CNNs and RNNs** help model n-grams and sequences in IR.
- **Transformers (BERT)** dominate modern NLP and IR applications.
- **Query expansion** leverages word embeddings for improved search results.

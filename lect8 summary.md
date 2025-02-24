## Problems
### 1. Long-Range Dependencies in N-Gram Models
- **Issue**: N-gram models struggle with long-distance word dependencies.
- **Solution**: Use recurrent neural networks (RNNs) and transformers for improved context modeling.
### 2. Vanishing and Exploding Gradients in RNNs
- **Issue**: Training deep RNNs can lead to unstable gradients.
- **Solution**: Use gated architectures like LSTMs and GRUs.
### 3. Computational Inefficiency of RNNs
- **Issue**: RNNs process sequences sequentially, making them slow and hard to parallelize.
- **Solution**: Use transformers with self-attention for parallelization.
### 4. Data Leakage in Training Transformer LMs
- **Issue**: Directly predicting the next word can cause information leakage.
- **Solution**: Use masked attention to prevent seeing future tokens.
## Solutions
### 1. Fixed-Window Neural Language Models
- **Concept**: Uses a fixed number of previous words to predict the next.
- **Formula**: 
	- $y=softmax(W_2h)$ 
	- $h=f(W_1x)$
	- where $x$ is the concatenated word embeddings:$x = [v_1,v_2,v_3,v_4]$
- **Pros**:
    - Reduces sparsity issues.
    - Model size is $O(n)$ instead of $O(\text{exp}(n))$.
- **Cons**:
    - Limited context window.
    - Cannot model long-range dependencies effectively.
### 2. Recurrent Neural Networks (RNNs)
- **Concept**: Uses hidden states to retain sequential information.
- **Formula**: $h_t = f(W_x x_t + W_h h_{t-1})$ where $h_t$ is the hidden state at time $t$.
- **Challenges**:
    - Vanishing/exploding gradients.
    - Hard to parallelize.
### 3. Transformer-Based Language Models
- **Concept**: Uses self-attention instead of recurrence.
- **Formula**: $\text{Attention}(Q,K,V) =\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- **Pros**:
    - Captures long-range dependencies.
    - Highly parallelizable.
    - Efficient for large datasets.
- **Cons**:
    - Quadratic complexity in sequence length.
### 4. Masked Attention for Autoregressive Training
- **Concept**: Prevents information leakage by masking future tokens.
- **Implementation**: Attention mask sets future token weights to $-\infty$ before softmax.
## Key Ideas
### 1. Tokenization for Neural LMs
- **Byte Pair Encoding (BPE)**: Merges frequent character pairs iteratively.
- **WordPiece**: Optimizes likelihood instead of frequency.
- **SentencePiece**: Works without whitespace-based segmentation.
### 2. Training Transformers for Language Modeling
- **Goal**: Predict the next token in a sequence.
- **Process**:
    - Shift input right to create labels.
    - Apply self-attention.
    - Compute loss over vocabulary.
    - Backpropagate to update parameters.
### 3. Self-Attention Mechanism
- **Advantage**: Captures contextual dependencies efficiently.
- **Formula**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- **Application**: Used in transformers like BERT and GPT.
### 4. Text Generation with Transformers
- **Approach**:
    - Use previous token output as the next input.
    - Adjust probabilities dynamically.
- **Example**:
    - Input: "The cat"
    - Generated: "The cat sat on the mat."
## Summary
- **N-Gram Models**: Limited context, sparsity issues.
- **RNNs**: Sequential processing, vanishing gradients.
- **Transformers**: Parallelized, self-attention-based, efficient for large datasets.
- **Masking**: Prevents data leakage in training.
- **Tokenization**: BPE, WordPiece, and SentencePiece optimize subword modeling.
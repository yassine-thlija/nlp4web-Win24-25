## Problems
### 1. Out-of-Vocabulary (OOV) Words
- **Issue**: Traditional word-level tokenization struggles with unseen words.
- **Solution**: Use subword tokenization methods like Byte Pair Encoding (BPE), WordPiece, or SentencePiece.
### 2. Zero Probability in N-Gram Models
- **Issue**: If an unseen n-gram appears, probability becomes zero.
- **Solution**: Apply smoothing techniques (Laplace/Add-One, Add-k, Backoff, Interpolation).
### 3. Computational Complexity in Large N-Grams
- **Issue**: Longer sequences have exponentially increasing probabilities to compute.
- **Solution**: Approximate using Markov assumption (bigram, trigram models).
## Solutions
### 1. N-Gram Language Models
- **Concept**: Assign probabilities to word sequences based on past occurrences.
- **Formula**: $P(w_1, w_2, ..., w_n) \approx \prod_{k=1}^{n} P(w_k | w_{k-1})$ (Bigram Model)
### 2. Byte Pair Encoding (BPE)
- **Concept**: Iteratively merges the most frequent character pairs into subword units.
- **Process**:
    - Initialize vocabulary with single characters.
    - Merge the most frequent adjacent pairs.
    - Repeat until the desired vocabulary size is reached.
### 3. WordPiece Tokenization
- **Concept**: Similar to BPE but selects pairs that maximize sequence likelihood.
- **Formula**: $\frac{P(ug)}{P(u)*P(g)}$ ensures merging improves probability.
### 4. Perplexity as a Model Evaluation Metric
- **Concept**: Measures uncertainty of a model in predicting a text sample.
- **Formula**: $Perplexity(W)=\sqrt[N]{\frac{1}{P(W)^{'}}}$ (for a bigram model)where $W$ is the sequence and $N$ is the number of words.
- General Perplexity Formula:
	- $Perplexity(W)=\huge{\sqrt[N]{\frac{1}{\prod_{i=1}^{N}P(w_i|w_{i-1})}}}$
### 5. Log Probability for Numerical Stability
- **Concept**: Reduces underflow issues when dealing with small probabilities.
- **Formula**: $P_1*P_2*P_3=\large{e^{(log⁡P_1+log⁡P_2+log⁡P_3)}}$
## Key Ideas
### 1. Why Language Models Are Useful
- **Applications**: Speech recognition, spelling correction, machine translation, summarization, question answering.
- **Example**: P(“I wanted to let her go”)>P(“I want it two letter go”)
### 2. Bigram Language Model Example
- **Example sentence**: "There once was a ship that put to sea."
- **Probability Calculation**: 
	- $P(w_n|w_{n-1})=\Large\frac{C(w_{n-1}w_n)}{C(w_{n-1})}$
### 3. Text Generation with Language Models
- **Use Case**: Predict next words in a sequence.
- **Example**:
    - Input: "I go to the"
    - Possible outputs with probabilities:
        - "cinema" (0.374)
        - "island" (0.0023)
        - "be" (0.000001)
## Summary
- **N-Gram Models**: Simple but struggle with OOV words and probability sparsity.
- **Tokenization Methods**: Subword approaches like BPE, WordPiece, and SentencePiece address OOV issues.
- **Evaluation**: Perplexity and log probability help compare models.
- **Generation**: Language models can generate text based on learned probabilities.
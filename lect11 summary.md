## Problems
### 1. Mismatch Between Pre-Training and User Intent
- **Issue**: LLMs are pre-trained on broad datasets but do not inherently perform user-specific tasks.
- **Solution**: Adaptation techniques like fine-tuning, instruction-tuning, and reinforcement learning.
### 2. Limits of Instruction-Tuning
- **Issue**: Requires large labeled datasets, is prone to rote memorization, and may encourage hallucinations.
- **Solution**: Reinforcement Learning from Human Feedback (RLHF) provides dynamic adaptation.
### 3. Bias in Language Models
- **Issue**: Models can be biased due to majority label bias and recency bias.
- **Solution**: Diversify prompts, balance demonstration distributions, and apply bias-mitigation strategies.
### 4. Handling Long Contexts
- **Issue**: Transformers have limited context windows, making long-document processing inefficient.
- **Solution**: Use sparse attention mechanisms, retrieval-augmented generation, and memory-efficient architectures.
## Solutions
### 1. Reinforcement Learning from Human Feedback (RLHF)
- **Concept**: Optimizes LLM behavior by training on ranked human preferences.
- **Process**:
    1. Collect human preferences on model outputs.
    2. Train a reward model to predict preference scores.
    3. Fine-tune the LLM using reinforcement learning.
- **Formula**: $\hat{\theta} = \arg\max_{\theta} \mathbb{E}_{s \sim p_\theta} R(s; \text{prompt})$
### 2. Instruction-Tuning
- **Concept**: Fine-tunes LLMs on (instruction, output) pairs to improve generalization.
- **Advantages**:
    - Enhances zero-shot and few-shot performance.
    - Reduces overfitting on narrow datasets.
- **Challenges**:
    - Requires diverse instruction data.
    - Sensitive to formatting and wording.
### 3. Sparse Attention for Long Contexts
- **Concept**: Reduces computational complexity by attending to selective tokens.
- **Methods**:
    - **Longformer** (sliding window attention)
    - **BigBird** (global-local attention)
    - **Reformer** (hash-based attention)
- **Benefit**: Enables efficient processing of documents beyond 2k tokens.
### 4. Retrieval-Augmented Generation (RAG)
- **Concept**: Enhances LLM responses by retrieving relevant external data.
- **Process**:
    1. Query an external datastore (e.g., Wikipedia, document corpora).
    2. Retrieve top-k relevant passages.
    3. Fuse retrieved content into the LLM generation pipeline.
- **Advantage**: Reduces reliance on outdated or incomplete model knowledge.
## Key Ideas
### 1. Regularization in RLHF
- **Problem**: LLMs may "game" reward functions, leading to nonsensical outputs.
- **Solution**: Add a penalty term to prevent divergence from pre-trained distributions: $\large{\hat{R}(s; p) = R(s; p) - \beta \log \frac{p_{RL}(s)}{p_{PT}(s)}}$ where $p_{RL}$ is the RL policy and $p_{PT}$ is the pre-trained model.
### 2. Bias in Reinforcement Learning
- **Issue**: Human feedback introduces bias due to subjective preferences.
- **Solution**:
    - Use diverse human annotators.
    - Normalize and debias reward functions.
### 3. Neural Retrieval for Augmenting LLMs
- **Concept**: LLMs retrieve relevant information rather than memorizing everything.
- **Approach**:
    - Use dense retrievers like **ColBERT** for document search.
    - Integrate neural re-ranking for more accurate retrieval.
### 4. Multi-Step Prompting for Improved Reasoning
- **Concept**: Chain-of-Thought (CoT) prompting improves multi-step reasoning.
- **Example**:
    - **Question**: "What is the capital of France?"
    - **CoT Prompting**: "France is a country in Europe. Its capital is..."
## Summary
- **Instruction-Tuning vs. RLHF**: RLHF dynamically adapts to human preferences, while instruction-tuning improves generalization.
- **Handling Long Contexts**: Sparse attention and retrieval-augmented models extend LLM capabilities.
- **Bias Mitigation**: Diverse annotation and normalization strategies reduce biases in feedback loops.
- **Future Directions**: Scaling RLHF, improving retrieval efficiency, and exploring hybrid approaches for LLM alignment.
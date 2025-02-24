## Problems
### 1. Mismatch Between Pre-Training and User Intent
- **Issue**: Large language models (LLMs) are pre-trained on broad datasets but do not inherently perform specific tasks.
- **Solution**: Adaptation techniques like fine-tuning and prompting.
### 2. Inefficiency of Full Model Fine-Tuning
- **Issue**: Updating all model parameters for each task is computationally expensive.
- **Solution**: Use parameter-efficient fine-tuning methods like adapters and BitFit.
### 3. Context-Length Limitations
- **Issue**: Transformer-based models have a fixed context window.
- **Solution**: Use retrieval-augmented generation (RAG) or memory-augmented architectures.
### 4. Training Instability in Transformers
- **Issue**: Large models can be unstable due to vanishing/exploding gradients or data leakage.
- **Solution**: Apply masked attention, layer normalization, and optimized training schedules.
## Solutions
### 1. Encoder-Decoder Transformer Architecture
- **Concept**: Separates encoding and decoding for better context retention.
- **Formula**:$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- **Pros**:
    - Supports sequence-to-sequence tasks (e.g., translation).
    - Enables bidirectional encoding and autoregressive decoding.
- **Cons**:
    - Higher computational complexity.
### 2. Fine-Tuning for Task-Specific Adaptation
- **Concept**: Updates model parameters based on task-specific data.
- **Types**:
    - **Whole-model fine-tuning**: Updates all parameters.
    - **Head tuning**: Updates only task-specific layers.
- **Example Applications**:
    - Translation
    - POS Tagging
    - Sentiment Classification
    - Language Modeling
### 3. Parameter-Efficient Fine-Tuning
- **Concept**: Modifies a small subset of parameters instead of the full model.
- **Methods**:
    - **Adapters**: Add small trainable layers between frozen transformer layers.
    - **BitFit**: Fine-tunes only bias terms in self-attention and MLP layers.
    - **Selective Methods**: Fine-tune a subset of layers based on importance.
- **Advantages**:
    - Reduces memory footprint.
    - Maintains generalization across tasks.
### 4. Attention Masking for Sequential Prediction
- **Concept**: Prevents models from attending to future tokens during training.
- **Implementation**:
    - Use a triangular mask that sets future token weights to $-\infty$ before softmax.
- **Benefit**: Enables autoregressive text generation without information leakage.
## Key Ideas
### 1. Adaptation Strategies for LLMs
- **Fine-Tuning**: Updates model weights for specific tasks.
- **Prompting**: Reformulates input text to elicit desired responses.
### 2. Self-Attention Mechanism in Transformers
- **Formula**:$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- **Application**:
    - Used in BERT for contextual word representations.
    - Used in GPT for autoregressive text generation.
### 3. BitFit: Bias-Only Fine-Tuning
- **Concept**: Only updates bias terms in transformer layers.
- **Advantage**: Reduces computational cost while maintaining effectiveness.
- **Tradeoff**: May not capture deep structural adjustments needed for some tasks.
### 4. Text Generation in Transformers
- **Approach**:
    - Use the previous token’s output as the next input.
    - Dynamically adjust probabilities at each step.
- **Example**:
    - Input: "The cat"
    - Generated: "The cat sat on the mat."
## Summary
- **Pre-training vs. Adaptation**: LLMs require fine-tuning or prompting to align with specific tasks.
- **Fine-Tuning Strategies**: Whole-model, head tuning, parameter-efficient tuning (adapters, BitFit).
- **Self-Attention**: Enables efficient and scalable contextual understanding.
- **Masking**: Ensures autoregressive generation without information leakage.
- **Text Generation**: Uses iterative token prediction for sequence completion.
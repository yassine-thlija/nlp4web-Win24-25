## Problems
### 1. Mismatch Between Pre-Training and User Intent
- **Issue**: Large language models (LLMs) are pre-trained on broad datasets but do not inherently perform specific tasks.
- **Solution**: Adaptation techniques like fine-tuning and instruction-tuning.
### 2. Sensitivity of In-Context Learning
- **Issue**: Model performance varies greatly depending on prompt format, demonstration order, and wording.
- **Solution**: Careful selection and optimization of prompts.
### 3. Bias in Language Models
- **Issue**: Majority label bias and recency bias can skew predictions.
- **Solution**: Diversify prompts and balance demonstration distributions.
### 4. Limitations in Reasoning Tasks
- **Issue**: LLMs struggle with multi-step reasoning without explicit fine-tuning.
- **Solution**: Use Chain-of-Thought (CoT) prompting and multi-step reasoning techniques.
## Solutions
### 1. In-Context Learning (ICL)
- **Concept**: LLMs generate responses by conditioning on provided examples without parameter updates.
- **Formula**: $P(y∣x,D)$ where $D$ is the in-context demonstration set.
- **Challenges**:
    - Highly sensitive to input formatting.
    - Prone to biases from examples.
### 2. Chain-of-Thought (CoT) Prompting
- **Concept**: Encourages models to generate intermediate reasoning steps before the final answer.
- **Example**:
    - **Question**: "Elon Musk"
    - **CoT Response**: "The last letter of 'Elon' is 'n'. The last letter of 'Musk' is 'k'. Concatenating 'n' and 'k' gives 'nk'."
### 3. Self-Consistency in Prompting
- **Concept**: Generates multiple responses to the same prompt and selects the most common answer.
- **Benefit**: Improves accuracy in reasoning tasks.
- **Example**:
    - Multiple reasoning paths → Aggregated final answer.
### 4. Instruction-Tuning
- **Concept**: Fine-tunes LLMs on a dataset of (instruction, output) pairs to improve task generalization.
- **Advantages**:
    - Enhances zero-shot performance.
    - More robust to variations in input phrasing.
## Key Ideas
### 1. Alignment in LLMs
- **Problem**: Pre-trained LMs do not always follow human values and ethical guidelines.
- **Solution**:
    - Reinforcement Learning from Human Feedback (RLHF).
    - Instruction tuning to guide behavior.
### 2. Bias in Language Models
- **Majority Label Bias**: Frequent labels dominate predictions.
- **Recency Bias**: Recent examples in prompts disproportionately influence model output.
### 3. Multi-Step Prompting
- **Approach**:
    - Use structured prompts to guide step-by-step reasoning.
    - Apply self-consistency to validate outputs.
### 4. Scaling Instruction-Tuning
- **Concept**: Increasing dataset size and task diversity improves LLM performance.
- **Finding**: Model accuracy improves linearly with instruction diversity and data volume.
## Summary
- **Pre-Training vs. Adaptation**: LLMs need instruction tuning and fine-tuning for real-world usability.
- **Prompting Strategies**: In-Context Learning, Chain-of-Thought, and Self-Consistency improve reasoning capabilities.
- **Bias Mitigation**: Adjusting demonstration selection reduces majority and recency biases.
- **Alignment**: Instruction-tuned models better follow human values and intent.
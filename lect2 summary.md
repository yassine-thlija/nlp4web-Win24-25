## Text Classification
### Approaches
- **Rule-Based Methods**:    
	- Uses handcrafted linguistic rules.
	- Pros: High precision.
	- Cons: Expensive to build and maintain.
- **Supervised Machine Learning**:
	- Requires labeled training data.
	- Uses feature extraction and classification models.
	- Pros: More accurate, easier to maintain.
	- Cons: Needs substantial training data.
### Naïve Bayes Classifier
- **Conditional Probability**:
    $P(O, E) = P(O) * P(E|O)$
- where:
	- $P(O)$ is the probability of outcome $O$.
	- $P(E∣O)$ is the probability of evidence $E$ given $O$.
- **Bayes' Rule**:
	- $P(O|E) = \frac{P(E|O) * P(O)}{P(E)}$
- Used to compute the most probable class given some evidence.
- **Naïve Bayes Simplification**:
- $P(O|E_1, ..., E_n) = \frac{P(E_1|O) * P(E_2|O) * ... * P(E_n|O) * P(O)}{P(E_1, E_2, ..., E_n)}$    
- Assumes independence of features for simpler computation :it assumption is violated too much it can lead to inaccurate results : resolved with proper feature selection.    
### Hidden Markov Models (HMM) for NLP
- **Sequence Labeling as Classification** : Standard classification problems assume decisions are mutually independant : not always the case in modern NLP(involve making many connected decisions , each resolves a different ambiguity but which are mutually dependant).
- Example :Sequence Labeling(like POS ,NER ...)
* Approach 1 : Sliding Window to integrate a context window as input for the classifier.
	* Even Better : include outputs of previous or next tokens(Forward/Backward Classification) as inputs for the classifier(however some tokens are better disambiguated with FW other with BW)
	* $\implies$Problems with this approach : difficult to integrate information from both sides/ difficult to propagate uncertainty of decisions and find a joint assignment of all labels of the sequence .
		- Solution : Probabilistic Sequence Models: integrating uncertainty over multiple interdependant classification and collectively determine the most likely global assignments)
		- Example : **Generative sequence models : like HMMs** 

- **Definition**: A statistical model where states are hidden, and outputs are observed.
- **Applications**:
	- Part-of-Speech (POS) Tagging.
	- Named Entity Recognition.
	- Speech Recognition.
- **HMM Components**:
	- **States**: Representing POS tags.
	- **Transition Probabilities**: Probability of moving from one state to another.
	- **Emission Probabilities**: Probability of an observed word given a hidden state.
- **Viterbi Algorithm for Decoding**:
- $argmax_{s_1, s_2, ..., s_T} P(O_1, ..., O_T, s_1, ..., s_T)$
- Used to find the most probable sequence of hidden states.
## Summary

- **Natural Language Processing (NLP)** involves extracting, analyzing, and understanding text.
- **Challenges** include ambiguity, noise, and context dependency.
- **Solutions** range from traditional linguistic methods to deep learning approaches.
- **Text classification** is a key NLP task, commonly solved using rule-based, supervised learning, and probabilistic models like Naïve Bayes.
- **Hidden Markov Models** are widely used for sequence-based NLP tasks such as POS tagging.
- **Mathematical models and AI techniques** play a crucial role in improving NLP systems.

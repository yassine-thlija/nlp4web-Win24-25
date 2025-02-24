## Introduction to Information Retrieval
### Key Concepts
- **Relevance:** Does a document satisfy the information need of the user and does it help complete the user’s task?$\implies$in our model (Inverted Index) we deal with relevance to the query rather than the need(counting terms in query and doc) the difference:
	- An information need is a user's underlying goal, while a query is the specific formulation of that need in a search system.
		- Example: 
			- Information need: "I want to find the best hiking trails near me.“
			- Query: "best hiking trails in Darmstadt."
- **Inverted Index**:
	- Allows efficient document retrieval from large collections.
	- Stores statistics per term needed for scoring models.
	- Components:
		- **Document Frequency ($df$)**: Number of documents containing a term.
		- **Term Frequency ($tf$)**: Frequency of a term in a document.
		- **Document Length**: Number of words in a document.
		- **Average Document Length**: Mean length of all documents in the collection.
- **Querying the Inverted Index**:
	- Only operates on term frequency statistics rather than full documents.
	- Uses relevance scoring models to rank documents.
- **Types of Queries**:
	- **Exact Matching**: Full-word matching.
	- **Boolean Queries**: AND/OR/NOT operations between words.
	- **Expanded Queries**: Synonyms and related words included automatically.
	- **Wildcard Queries**: Using placeholders for unknown characters.
	- **Phrase Queries**: Search for exact sequences of words.
## Relevance Scoring Models
### TF-IDF (Term Frequency - Inverse Document Frequency)
- **Formula**:
- $TF-IDF(q, d) = \large{\sum_\limits{t\in T_d \cap T_q}}\log(1 + tf_{t,d}) * log(\frac{|D|}{df_t})$
	- where:
		- $tf_{t,d}$ = term frequency of term $t$ in document $d$
		- $D$ = total number of documents
		- $df_t$ = number of documents containing term $t$
- **Key Properties**:
	- Rare terms are given higher weights.
	- High term frequency in a document increases its relevance.
	- Logarithm is used to dampen extreme frequency values.

### BM25 (Best Matching 25)
- **Formula**:
	- $BM25(q, d) = \large{\sum_\limits{t\in T_d \cap T_q}} \Large{\frac{tf_{t,d}}{k_1((1-b)+b\cdot\frac{dl_d}{avgdl}+tf_{t,d})} * \log\frac{|D| - df_t + 0.5}{df_t + 0.5}}$
	- where:
		- $k1$, $b$ are hyperparameters controlling term frequency scaling and document length normalization.
		- $dl_d$ = document length of $d$
		- $avgdl$ = average document length in the collection
- **Advantages Over TF-IDF**:
	- Better handling of term frequency saturation.
	- Adjusts for document length normalization.
	- Used widely in modern search engines.

## Evaluation Metrics for IR Systems
### Precision and Recall

- **Precision**: Fraction of retrieved documents that are relevant.
	- Precision = $\frac{TP}{TP + FP}$
- **Recall**: Fraction of relevant documents retrieved.
	- Recall = $\frac{TP}{TP + FN}$
- **F1 Score**: Harmonic mean of precision and recall.
	- $F1$ = $\frac{2\cdot(Precision\cdot Recall)}{Precision + Recall}$
### Mean Reciprocal Rank (MRR)
- Measures how quickly a system retrieves the first relevant document.
	- $MRR$ = $\frac{1}{|Q|}\cdot \sum \frac{1}{rank_i}$
	- where $rank_i$ is the position of the first relevant document for query $i$.
### Mean Average Precision (MAP)
- Computes the mean precision across all relevant documents.
	- $MAP$ = $\frac{1}{|Q|}\cdot \sum AP_i$
	- where $AP_i$ is the average precision for query ii.

### Normalized Discounted Cumulative Gain (nDCG)
- **Formula**:
    $DCG$ = $\sum\frac{rel_i}{log_2(i + 1)}$
    $nDCG$ = $\frac{DCG}{IDCG}$
    where:
	- $rel_i$ = relevance score at position $i$
	- $IDCG$ = ideal $DCG$ (best possible ranking)
- **Key Benefit**: Considers graded relevance, not just binary.
## Summary
- **Inverted Index** is essential for efficient document retrieval.
- **$TF-IDF$ and $BM25$** are core relevance ranking models.
- **Precision, Recall, $MRR$, $MAP$, and $nDCG$** evaluate the effectiveness of retrieval models.
- **$BM25$ outperforms $TF-IDF$** in most modern search applications.

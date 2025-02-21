from jet.logger import logger
from jet.transformers.formatters import format_json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample documents
documents = [
    "React Native is a framework for building mobile apps.",
    "Flutter and Swift are alternatives to React Native.",
    "React is a JavaScript library for building UIs.",
    "Node.js is used for backend development.",
]

# Convert documents to lowercase tokens
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Query handling
query_terms = ["React"]
negation_terms = ["React Native"]

# BM25 ranking
bm25_scores = bm25.get_scores(query_terms)

# Filter documents **excluding** "React Native"
valid_indices = [
    i for i, doc in enumerate(documents)
    if not any(term.lower() in doc.lower() for term in negation_terms)
]

# Load Sentence Transformer Model
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L12-v2')

# Vectorize only the filtered documents
filtered_vectors = np.array([model.encode(documents[i])
                            for i in valid_indices])
query_vector = model.encode("React")

# Compute similarity on filtered results
similarities = cosine_similarity([query_vector], filtered_vectors)[0]
sorted_indices = np.argsort(similarities)[::-1]

# Display final results
final_results = [documents[valid_indices[i]] for i in sorted_indices]

logger.newline()
logger.info(f"BM25 + Vector Search Results ({len(final_results)}):")
logger.debug(f"(Queries: {query_terms})")
logger.debug(f"(Excludes: {negation_terms})")
logger.success(format_json(final_results))

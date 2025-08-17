from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np

# Load models
jina_model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
jina_model.max_seq_length = 1024  # Adjust as needed
st_model = SentenceTransformer(
    "sentence-transformers/static-retrieval-mrl-en-v1")

# Sample texts
texts = ["How is the weather today?",
         "What is the current weather like today?"]

# Generate embeddings
jina_embeddings = jina_model.encode(texts, normalize_embeddings=True)
st_embeddings = st_model.encode(texts, normalize_embeddings=True)

# Compute cosine similarity
jina_sim = cos_sim(jina_embeddings[0], jina_embeddings[1]).item()
st_sim = cos_sim(st_embeddings[0], st_embeddings[1]).item()

print(f"Jina Embeddings Similarity: {jina_sim:.4f}")
print(f"Sentence Transformers Similarity: {st_sim:.4f}")

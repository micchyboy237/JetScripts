# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

import torch
from sentence_transformers import SentenceTransformer

# Check for MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Manually move the underlying model to MPS
model.to(device)

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents (will run on MPS)
query_embeddings = model.encode(queries, prompt_name="query", device=device)
document_embeddings = model.encode(documents, device=device)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
# Example output:
# tensor([[0.7646, 0.1414],
#         [0.1355, 0.6000]])

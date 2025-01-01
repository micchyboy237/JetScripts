from sentence_transformers import SentenceTransformer, util

# Sample list of package module paths
module_paths = [
    "numpy.linalg.linalg",
    "numpy.core.multiarray",
    "pandas.core.frame",
    "matplotlib.pyplot",
    "sklearn.linear_model",
    "torch.nn.functional",
]

# Multiline import argument (query)
import_arg = """
from sklearn.linear_model import LogisticRegression
from numpy.linalg import inv
import torch
"""

# Load a pre-trained model from sentence-transformers
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode the module paths and the query
module_embeddings = model.encode(module_paths, convert_to_tensor=True)
query_embedding = model.encode(import_arg, convert_to_tensor=True)

# Compute cosine similarities
scores = util.cos_sim(query_embedding, module_embeddings)[0].cpu().numpy()

# Rank the module paths by similarity score
ranked_results = sorted(zip(module_paths, scores),
                        key=lambda x: x[1], reverse=True)

# Display results
print("Top relevant modules:")
for path, score in ranked_results:
    print(f"{path}: {score:.4f}")

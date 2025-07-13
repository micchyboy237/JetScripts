from jet.models.model_registry.transformers.cross_encoder_model_registry import CrossEncoderRegistry

model = CrossEncoderRegistry.load_model('cross-encoder/ms-marco-MiniLM-L6-v2')

module_paths = [
    "numpy.linalg.linalg",
    "numpy.core.multiarray",
    "pandas.core.frame",
    "matplotlib.pyplot",
    "sklearn.linear_model",
    "torch.nn.functional",
]

queries = """import matplotlib.pyplot as plt
from numpy.linalg import inv
import torch"""

# Split the queries by new lines
query_list = queries.split('\n')

# Create pairs of each query with each module path
pairs = [(query, path) for query in query_list for path in module_paths]

# Compute relevance scores
scores = model.predict(pairs)
ranked_results = sorted(zip(module_paths, scores),
                        key=lambda x: x[1], reverse=True)
print("Results:", ranked_results)

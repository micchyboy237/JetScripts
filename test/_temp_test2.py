from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings

# 1. Specify preffered dimensions
dimensions = 512

# 2. load model
model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)

# The prompt used for query retrieval tasks:
# query_prompt = 'Represent this sentence for searching relevant passages: '

query = "A man is eating a piece of bread"
docs = [
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]

# 2. Encode
query_embedding = model.encode(query, prompt_name="query")
# Equivalent Alternatives:
# query_embedding = model.encode(query_prompt + query)
# query_embedding = model.encode(query, prompt=query_prompt)

docs_embeddings = model.encode(docs)

# Optional: Quantize the embeddings
binary_query_embedding = quantize_embeddings(
    query_embedding, precision="ubinary")
binary_docs_embeddings = quantize_embeddings(
    docs_embeddings, precision="ubinary")

similarities = cos_sim(query_embedding, docs_embeddings)
print('similarities:', similarities)

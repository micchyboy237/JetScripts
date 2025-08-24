from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry


# model = SentenceTransformer(
#     "jinaai/jina-embeddings-v2-base-en",  # switch to en/zh for English or Chinese
#     trust_remote_code=True,
#     device="mps",
# )

model = SentenceTransformerRegistry.load_model(
    # "jina-embeddings-v2-base",
    "jina-embeddings-v2-small",
)

# control your input sequence length up to 8192
model.max_seq_length = 1024

embeddings = model.encode([
    'How is the weather today?',
    'What is the current weather like today?'
])
print(cos_sim(embeddings[0], embeddings[1]))

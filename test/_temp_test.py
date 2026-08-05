from unstructured.embed.openai import OpenAIEmbeddingConfig, embed

embedded_chunks = embed(
    elements=chunks,
    config=OpenAIEmbeddingConfig(
        model_name="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    ),
)

# Each element now has .embeddings attribute
vector = embedded_chunks[0].embeddings
print(f"Embedding dimension: {len(vector)}, first 3 values: {vector[:3]}")

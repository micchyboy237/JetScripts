from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Initialize the Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Create a collection
client.create_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=4, distance=Distance.DOT),
)

# Insert vectors into the collection
vectors = [
    [0.1, 0.2, 0.3, 0.4],
    [0.2, 0.3, 0.4, 0.5],
    [0.3, 0.4, 0.5, 0.6],
]
payloads = [{"id": i} for i in range(len(vectors))]

client.upsert(
    collection_name="test_collection",
    points=[
        {"id": i, "vector": vector, "payload": payload}
        for i, (vector, payload) in enumerate(zip(vectors, payloads))
    ]
)

# Perform a search
search_vector = [0.15, 0.25, 0.35, 0.45]
results = client.search(
    collection_name="test_collection",
    query_vector=search_vector,
    limit=3
)

print("Search results:", results)

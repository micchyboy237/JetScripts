import json
from jet.db.chroma import ChromaClient
from jet.llm.ollama import get_embedding_function, OllamaEmbeddingFunction
from jet.transformers import make_serializable
from jet.logger import logger

if __name__ == "__main__":
    embedding_func = OllamaEmbeddingFunction()
    collection_name = "example_collection"

    initial_data = [
        {
            "id": "item1",
            "document": "This is a common document.",
            "metadata": {"category": "similar"},
        },
        {
            "id": "item2",
            "document": "Different from others.",
            "metadata": {"category": "unique"},
        },
        {
            "id": "item3",
            "document": "Another sample common document.",
            "metadata": {"category": "similar"},
        },
        {
            "id": "item4",
            "document": "Unique item contents.",
            "metadata": {"category": "unique"},
        },
    ]

    client = ChromaClient(
        collection_name=collection_name,
        initial_data=initial_data,
        metadata={"description": "An example collection."},
        overwrite=True,
        embedding_function=embedding_func,
    )

    print("Collection created with initial data.")

    results = client.query()
    logger.debug("RESULTS:")
    logger.success(json.dumps(make_serializable(results), indent=2))

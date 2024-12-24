import json
from jet.db.chroma import ChromaClient
from jet.llm.ollama import OllamaEmbeddingFunction
from jet.transformers import make_serializable
from jet.logger import logger


def main_get():
    embedding_func = OllamaEmbeddingFunction()
    collection_name = "example_collection"

    client = ChromaClient(collection_name, embedding_func)

    calls = [
        {
            "type": "get_all",
        },
        {
            "type": "get_by_ids",
            "params": {
                "ids": ["item2", "item3"]
            }
        },
    ]

    for call in calls:
        logger.newline()
        logger.debug(f"Type: {call['type']}")
        results = client.get(**call.get("params", {}))
        logger.debug(f"Results for {call['type']}:")
        logger.success(json.dumps(make_serializable(results), indent=2))


def main_query():
    embedding_func = OllamaEmbeddingFunction()
    collection_name = "example_collection"

    client = ChromaClient(collection_name, embedding_func)

    # Example calls
    calls = [
        {
            "type": "texts",
            "params": {
                "texts": ["sample document"],
                "top_n": 3
            }
        },
        {
            "type": "embeddings",
            "params": {
                "embeddings": embedding_func(["sample document"]),
                "top_n": 3
            }
        },
        {
            "type": "where_filter",
            "params": {
                "texts": ["Another"],
                "top_n": 3,
                "where": {"category": "unique"}
            }
        },
        {
            "type": "where_document",
            "params": {
                "texts": ["sample"],
                "where_document": {"$contains": "Another"},
                "top_n": 3,
            }
        },
        {
            "type": "include",
            "params": {
                "texts": ["sample"],
                "where_document": {"$contains": "Another"},
                "include": ["documents", "distances", "metadatas", "uris", "data"],
                "top_n": 3,
            }
        },
    ]

    for call in calls:
        logger.newline()
        logger.debug(f"Type: {call['type']}")
        results = client.query(**call.get("params", {}))
        logger.debug(f"Results for {call['type']}:")
        logger.success(json.dumps(make_serializable(results), indent=2))


if __name__ == "__main__":
    logger.newline()
    logger.info("main_get()...")
    main_get()

    logger.newline()
    logger.info("main_query()...")
    main_query()

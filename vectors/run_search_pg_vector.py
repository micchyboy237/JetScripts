from typing import TypedDict
from jet.data.utils import generate_key
from jet.db.pgvector import PgVectorClient
from jet.db.pgvector.config import (
    DEFAULT_USER,
    DEFAULT_PASSWORD,
    DEFAULT_HOST,
    DEFAULT_PORT,
)
from jet.file.utils import load_file
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import logger
from jet.utils.commands import copy_to_clipboard
from shared.data_types.job import JobData


class VectorsWithId(TypedDict):
    id: str
    embedding: list[float]
    text: str


class SearchResult(TypedDict):
    id: str
    score: float
    text: str


def embed_text(text: str) -> list[float] | list[list[float]]:
    embed_func = get_embedding_function("mxbai-embed-large")
    embed_result = embed_func(text)
    return embed_result


if __name__ == '__main__':
    # model = "mxbai-embed-large"
    model = "nomic-embed-text"

    dbname = "jobs_db1"
    tablename = "embeddings"
    vector_dim = OLLAMA_MODEL_EMBEDDING_TOKENS[model]

    query = "React Native,Firebase"
    top_k = 10

    vectors_with_ids: list[VectorsWithId] = load_file(
        "generated/job-embeddings.json")

    vectors_with_ids_text_dict: dict[str, str] = {
        item["id"]: item["text"] for item in vectors_with_ids
    }

    query_vector = embed_text(query)

    logger.newline()
    logger.info("Searching...")
    logger.debug(query)

    with PgVectorClient(
        dbname=dbname,
        user=DEFAULT_USER,
        password=DEFAULT_PASSWORD,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
    ) as client:
        try:
            similar_vectors = client.search_similar(
                tablename, query_vector, top_k=top_k)

            results: list[SearchResult] = [
                {**item, "text": vectors_with_ids_text_dict[item["id"]]}
                for item in similar_vectors
            ]
            copy_to_clipboard(results)
            logger.newline()
            logger.debug(f"Top {top_k} similar vectors:")
            logger.success(results)

        except Exception as e:
            logger.newline()
            logger.error(f"Transaction failed:\n{e}")

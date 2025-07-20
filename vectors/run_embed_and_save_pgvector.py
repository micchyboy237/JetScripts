from typing import Any, TypedDict
from jet.db.postgres.pgvector import PgVectorClient
from jet.db.postgres.pgvector.utils import create_db, delete_db
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.llm.ollama.base import OllamaEmbedding
from jet.llm.utils.embeddings import get_embedding_function, get_ollama_embedding_function
from jet.logger import logger, time_it
from jet.token.token_utils import get_token_counts_info, token_counter
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.utils.object import extract_values_by_paths
from tqdm import tqdm
from jet.file.utils import load_file, save_file
from shared.data_types.job import JobData, JobEntities


class Vectors(TypedDict):
    id: str
    tokens: int
    text: str
    metadata: Any
    embedding: list[float]


class SavedVectors(TypedDict):
    model: str
    db_name: str
    collection_name: str
    vectors: list[Vectors]


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    embed_func = get_embedding_function(model)
    embed_results = embed_func(texts)
    return embed_results


if __name__ == '__main__':
    model = "mxbai-embed-large"
    # model = "nomic-embed-text"

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    jobs: list[JobData] = load_file(data_file)

    texts = []
    metadata = []

    json_attributes = [
        # "id",
        "title",
        "entities.technology_stack",
        "details",
        "tags",
    ]
    metadata_attributes = [
        "id",
        "title",
        "link",
        "company",
        "posted_date",
        "salary",
        "job_type",
        "hours_per_week",
        "domain",
        "tags",
        "keywords",
        "entities.role",
        "entities.application",
        "entities.technology_stack",
        "entities.qualifications",
    ]

    for item in tqdm(jobs):
        json_parts_dict = extract_values_by_paths(
            item, json_attributes, is_flattened=True) if json_attributes else None
        text_parts = []
        for key, value in json_parts_dict.items():
            value_str = str(value)
            if isinstance(value, list):
                value_str = ", ".join(value)
            if value_str.strip():
                text_parts.append(
                    f"{key.title().replace('_', ' ')}: {value_str}")

        text_content = "\n".join(text_parts) if text_parts else ""
        texts.append(text_content)

    for item in tqdm(jobs):
        metadata_parts_dict = extract_values_by_paths(
            item, metadata_attributes, is_flattened=True) if metadata_attributes else None
        metadata.append(metadata_parts_dict)

    texts = list(texts)
    metadata = list(metadata)

    if len(texts) != len(jobs) and len(metadata) != len(jobs):
        message = f"Text and metadata should match jobs count ({len(jobs)}): {len(texts)} | metadata: {len(metadata)} "
        logger.error(message)
        raise message

    token_counts_info = get_token_counts_info(texts, model)
    tokenized_data = token_counts_info["results"]

    copy_to_clipboard(token_counts_info)

    logger.debug(f"Jobs ({len(jobs)})")
    logger.success(format_json({
        "smallest": token_counts_info["min"],
        "largest": token_counts_info["max"],
    }))

    # Embed texts
    embed_results = embed_texts(texts, model)
    logger.debug(f"Embed Results: {len(embed_results)}")
    logger.success(f"Embeddings Dim: {len(embed_results[0])}")

    # Save embeddings
    vectors_with_ids: list[Vectors] = [
        {
            "id": jobs[idx]["id"],
            "tokens": tokenized_data[idx]["tokens"],
            "text": text,
            "metadata": metadata[idx],
            "embedding": embed_results[idx],
        }
        for idx, text in enumerate(texts)
    ]
    vectors_with_ids_dict = {item["id"]: item for item in vectors_with_ids}
    vectors_with_ids = list(vectors_with_ids_dict.values())

    dbname = "job_vectors_db1"
    tablename = "embeddings"
    vector_dim = OLLAMA_MODEL_EMBEDDING_TOKENS[model]

    delete_db(dbname)
    logger.debug(f"Dropped DB: {dbname}")

    create_db(dbname)
    logger.debug(f"Created DB: {dbname}")

    saved_vectors: SavedVectors = {
        "model": model,
        "db_name": dbname,
        "collection_name": tablename,
        "vectors": vectors_with_ids
    }
    save_file(saved_vectors, "generated/job_vectors_db1/job-embeddings.json")

    logger.newline()
    logger.info(
        f"Saving embeddings ({len(vectors_with_ids)})...")
    logger.debug(f"Vector Dim ({vector_dim})")

    with PgVectorClient(
        dbname=dbname,
        user="jethroestrada",
        password="",
        host="localhost",
        port=5432
    ) as client:
        try:
            client.create_table(tablename, vector_dim)

            # Insert multiple vectors
            vector_data: dict[str, list[float]] = {
                item["id"]: item["embedding"] for item in vectors_with_ids}
            client.insert_embeddings_by_ids(tablename, vector_data)
        except Exception as e:
            logger.newline()
            logger.error(f"Embed and save failed:")
            raise e

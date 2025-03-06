from typing import Any, TypedDict
import numpy as np

from jet.data.utils import generate_key
from jet.db.pgvector import PgVectorClient
from jet.db.pgvector.config import (
    DEFAULT_USER,
    DEFAULT_PASSWORD,
    DEFAULT_HOST,
    DEFAULT_PORT,
)
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


class VectorsWithId(TypedDict):
    id: str
    embedding: list[float]
    text: str
    metadata: Any


class LoadedVectors(TypedDict):
    model: str
    db_name: str
    collection_name: str
    vectors: list[VectorsWithId]


if __name__ == '__main__':

    dbname = "job_vectors_db1"
    tablename = "embeddings"

    loaded_vectors: LoadedVectors = load_file(
        "generated/job_vectors_db1/job-embeddings.json")
    vectors_with_ids = loaded_vectors["vectors"]
    model = loaded_vectors["model"]
    vector_dim = OLLAMA_MODEL_EMBEDDING_TOKENS[model]

    logger.info(f"Saving {len(vectors_with_ids)} embeddings...")

    with PgVectorClient(
        dbname=dbname,
        user=DEFAULT_USER,
        password=DEFAULT_PASSWORD,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
    ) as client:
        try:
            # Clear data
            client.delete_all_tables()

            client.create_table(tablename, vector_dim)

            # Insert multiple vectors with predefined IDs
            vector_data: dict[str, list[float]] = {
                item["id"]: item["embedding"] for item in vectors_with_ids}
            client.insert_vector_by_ids(tablename, vector_data)

            all_items = client.get_vectors(tablename)
            logger.success(f"Done saving {len(all_items)} embeddings!")

        except Exception as e:
            logger.newline()
            logger.error(f"Load and save failed:")
            raise e

import numpy as np

from jet.data.utils import generate_key
from jet.db.pgvector import PgVectorClient
from jet.db.pgvector.utils import create_db, delete_db
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


def embed_texts(texts: list[str]) -> list[list[float]]:
    # embed_model = OllamaEmbedding(model_name="mxbai-embed-large")
    # embed_results = embed_model.embed(texts)
    # embedding_function = get_ollama_embedding_function(
    #     model="nomic-embed-text"
    # )
    # embed_results = embedding_function(texts)
    embed_func = get_embedding_function("mxbai-embed-large")
    embed_results = embed_func(texts)
    return embed_results


if __name__ == '__main__':
    model = "mxbai-embed-large"
    # model = "nomic-embed-text"
    dbname = "jobs_db1"
    tablename = "embeddings"
    vector_dim = OLLAMA_MODEL_EMBEDDING_TOKENS[model]

    vectors_with_ids: dict[str, list[float]] = load_file(
        "generated/job-embeddings.json")

    # delete_db(dbname)
    # logger.debug(f"Dropped DB: {dbname}")

    # create_db(dbname)
    # logger.debug(f"Created DB: {dbname}")

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
            client.insert_vector_by_ids(tablename, vectors_with_ids)

            all_items = client.get_vectors(tablename)
            logger.success(f"Done saving {len(all_items)} embeddings!")

        except Exception as e:
            logger.newline()
            logger.error(f"Transaction failed:\n{e}")

from jet.data.utils import generate_key
from jet.db.pgvector import PgVectorClient
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

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    jobs: JobData = load_file(data_file)

    texts = []

    json_attributes = [
        "title",
        "keywords",
        "tags",
        # "entities.role",
        # "entities.application",
        # "entities.coding_libraries",
        # "entities.qualifications",
        "details",
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

    token_counts_info = get_token_counts_info(texts, model)
    copy_to_clipboard(token_counts_info)

    logger.debug(f"Jobs ({len(jobs)})")
    logger.success(format_json({
        "largest": token_counts_info["max"]["tokens"],
        "smallest": token_counts_info["min"]["tokens"],
        "total": token_counts_info["total"],
    }))

    # Embed texts
    embed_results = embed_texts(texts)
    logger.debug(f"Embed Results: {len(embed_results)}")
    logger.success(f"Embeddings Dim: {len(embed_results[0])}")

    # Save embeddings

    dbname = "jobs_db1"
    tablename = "embeddings"
    vector_dim = OLLAMA_MODEL_EMBEDDING_TOKENS[model]

    vectors_with_ids = {generate_key(
        text): embed_results[idx] for idx, text in enumerate(texts)}

    save_file(vectors_with_ids, "generated/job-embeddings.json")

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
            client.insert_vector_by_ids(tablename, vectors_with_ids)
        except Exception as e:
            logger.error(f"Transaction failed:\n{e}")

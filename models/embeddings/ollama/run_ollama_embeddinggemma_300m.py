from jet.llm.ollama.base import OllamaEmbedding
from jet.llm.utils.embeddings import get_embedding_function, get_ollama_embedding_function
from jet.logger import logger, time_it
from jet._token.token_utils import get_token_counts_info, token_counter
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.utils.object import extract_values_by_paths
from tqdm import tqdm
from jet.file.utils import load_file, save_file
from shared.data_types.job import JobData, JobEntities


@time_it
def embed_texts(texts: list[str], model: str = "embeddinggemma-300m") -> list[list[float]]:
    embed_func = get_embedding_function(model)
    embed_results = embed_func(texts)
    return embed_results


if __name__ == '__main__':
    model = "embeddinggemma-300m"

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/saved/jobs.json"
    jobs: JobData = load_file(data_file)
    jobs = jobs[:10]

    texts = []

    json_attributes = [
        "title",
        "keywords",
        "tags",
        # "entities.role",
        # "entities.application",
        # "entities.technology_stack",
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
        "average": token_counts_info["average"],
        "largest": token_counts_info["max"],
        "smallest": token_counts_info["min"],
    }))

    embed_results = embed_texts(texts, model)
    logger.debug(f"Embed Results: {len(embed_results)}")
    logger.success(f"Embeddings Dim: {len(embed_results[0])}")

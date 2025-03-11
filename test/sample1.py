from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.logger import logger
from jet.token.token_utils import split_texts, token_counter
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.file.utils import load_file, save_file


if __name__ == "__main__":
    llm_model = "llama3.1"
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/langchain/cookbook/generated/RAPTOR/clusters.json"

    # max_tokens = 0.5
    overlap = 100
    model_max_length = OLLAMA_MODEL_EMBEDDING_TOKENS[llm_model]

    data: list = load_file(data_file)
    data = sorted(data, key=lambda x: len(x["text"]), reverse=True)
    largest_item = data[0]

    largest_token_count = token_counter(largest_item["text"], llm_model)
    logger.log("largest_token_count:", largest_token_count,
               colors=["GRAY", "DEBUG"])

    chunk_size = 200
    logger.log("chunk_size:", chunk_size,
               colors=["GRAY", "DEBUG"])

    splitted_texts = split_texts(
        largest_item["text"], llm_model, chunk_size, overlap)
    splitted_token_counts: list[int] = token_counter(
        splitted_texts, llm_model, prevent_total=True)

    copy_to_clipboard(splitted_texts)
    logger.success(format_json(splitted_texts))
    logger.success(format_json(splitted_token_counts))

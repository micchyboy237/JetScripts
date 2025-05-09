

import os
from pathlib import Path

from jet.code.utils import ProcessedResult, preprocess_notebooks_to_markdowns
from jet.file.utils import load_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import ModelKey
from jet.llm.mlx.token_utils import chunk_text
from jet.llm.mlx.utils import get_model_max_tokens
from jet.logger import logger

MODEL: ModelKey = "llama-3.2-1b-instruct-4bit"

SYSTEM_PROMPT = """
"""

PROMPT_TEMPLATE = """
"""

mlx = MLX(MODEL)


def call_llm_generation(context: str) -> str:
    response = ""
    for chunk in mlx.stream_chat(context, max_tokens=-1):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return response


def preprocess_data(data: list[ProcessedResult]) -> str:
    model_max_tokens = get_model_max_tokens(MODEL)
    texts = [d["text"] for d in data]
    chunk_size = 300
    chunks: list[str] = chunk_text(texts, chunk_size)
    context = "\n\n".join(chunks)
    return context


def run_generate_chunker_class(json_file_or_dir):
    # Process the markdown file
    if os.path.isdir(json_file_or_dir):
        for file in Path(json_file_or_dir).glob('*.json'):
            data: list[ProcessedResult] = load_file(str(file))
            preprocessed_data = preprocess_data(data)
            call_llm_generation(preprocessed_data)
    else:
        data: list[ProcessedResult] = load_file(json_file_or_dir)
        preprocessed_data = preprocess_data(data)
        call_llm_generation(preprocessed_data)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "generated", "markdown_processing")

    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/docs"
    preprocess_notebooks_to_markdowns(md_file, output_dir)

    run_generate_chunker_class(output_dir)

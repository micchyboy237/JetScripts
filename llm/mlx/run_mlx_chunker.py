

import os
from pathlib import Path
from typing import Generator, TypedDict, Union

from jet.code.utils import ProcessedResult, preprocess_notebooks_to_markdowns
from jet.file.utils import load_file, save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import ModelKey
from jet.llm.mlx.token_utils import chunk_text, merge_texts
from jet.llm.mlx.utils import get_model_max_tokens
from jet.logger import logger

MODEL: ModelKey = "llama-3.2-3b-instruct-4bit"

SYSTEM_PROMPT = """
You are a code generation assistant. Given relevant code context, generate a Python class named Chunker that splits input data into logical chunks. Focus on clean, efficient, and well-documented code. Output only the class definition.
"""


PROMPT_TEMPLATE = """
"""

mlx = MLX(MODEL)


def call_llm_generation(contexts: list[str]) -> str:
    context = "\n\n".join(contexts)  # Filter by model max tokens
    model_max_tokens = get_model_max_tokens(mlx.model_path)
    max_tokens = model_max_tokens - 640
    merged_texts = merge_texts(
        context, mlx.tokenizer, max_length=max_tokens)

    current_text = ""
    current_token_count = 0

    for text, token_count in zip(merged_texts["texts"], merged_texts["token_counts"]):
        if current_token_count + token_count > max_tokens:
            break
        current_text += text
        current_token_count += token_count

    response = ""
    for chunk in mlx.stream_chat(current_text, max_tokens=-1, system_prompt=SYSTEM_PROMPT):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return response


def preprocess_data(data: list[ProcessedResult]) -> list[str]:
    texts = [d["text"] for d in data]
    chunk_size = 300
    chunks = chunk_text(texts, chunk_size)
    return chunks


class GenerationResult(TypedDict):
    source: Union[str, Path]
    code: str


def run_generate_chunker_class(json_file_or_dir) -> Generator[GenerationResult, None, None]:
    # Process the markdown file
    if os.path.isdir(json_file_or_dir):
        for file in Path(json_file_or_dir).glob('*.json'):
            data: list[ProcessedResult] = load_file(str(file))
            preprocessed_data = preprocess_data(data)
            response = call_llm_generation(preprocessed_data)
            yield {"source": file, "code": response}
    else:
        data: list[ProcessedResult] = load_file(json_file_or_dir)
        preprocessed_data = preprocess_data(data)
        response = call_llm_generation(preprocessed_data)
        yield {"source": json_file_or_dir, "code": response}


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0], "markdown_processing")

    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/docs"

    preprocess_notebooks_to_markdowns(md_file, output_dir)

    results = run_generate_chunker_class(output_dir)

    chunker_classes_output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0], "chunker_classes")

    for result in results:
        file_no_ext = os.path.splitext(os.path.basename(result['source']))[0]
        file = f"{file_no_ext}.py"
        save_file(result["code"], f"{chunker_classes_output_dir}/{file}")

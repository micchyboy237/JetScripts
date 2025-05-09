

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


class GenerationResult(TypedDict):
    context: str
    response: str


def call_llm_generation(texts: list[str]) -> GenerationResult:
    context = mlx.filter_docs(texts)

    response = ""
    for chunk in mlx.stream_chat(context, max_tokens=-1, system_prompt=SYSTEM_PROMPT):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return {
        "context": context,
        "response": response,
    }


def preprocess_data(data: list[ProcessedResult]) -> list[str]:
    texts = [d["text"] for d in data]
    chunk_size = 300
    chunks = chunk_text(texts, chunk_size)
    return chunks


class GeneratorGenerationResult(TypedDict):
    source: Union[str, Path]
    code: str
    data: list[str]


def run_generate_chunker_class(json_file_or_dir) -> Generator[GeneratorGenerationResult, None, None]:
    # Process the markdown file
    if os.path.isdir(json_file_or_dir):
        for file in Path(json_file_or_dir).glob('*.json'):
            logger.info(f"\nProcessing {file}")
            data: list[ProcessedResult] = load_file(str(file))
            preprocessed_data = preprocess_data(data)
            result = call_llm_generation(preprocessed_data)
            yield {"source": file, "code": result["response"], "data": preprocessed_data}
    else:
        logger.info(f"\nProcessing {json_file_or_dir}")
        data: list[ProcessedResult] = load_file(json_file_or_dir)
        preprocessed_data = preprocess_data(data)
        result = call_llm_generation(preprocessed_data)
        yield {"source": json_file_or_dir, "code": result["response"], "data": preprocessed_data}


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0], "markdown_processing")

    md_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/docs"

    preprocess_notebooks_to_markdowns(md_dir, output_dir)

    results = run_generate_chunker_class(output_dir)

    chunker_classes_output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0], "chunker_classes")

    for result in results:
        file_no_ext = os.path.splitext(os.path.basename(result['source']))[0]
        sub_dir = f"{chunker_classes_output_dir}/{file_no_ext}"
        logger.newline()
        save_file(result["code"], f"{sub_dir}/chunker.py")
        save_file(result["data"], f"{sub_dir}/data.json")

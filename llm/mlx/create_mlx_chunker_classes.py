

import os
from pathlib import Path
from typing import Generator, TypedDict, Union

from jet.code.utils import ProcessedResult, preprocess_notebooks_to_markdowns
from jet.features.rag_llm_generation import SimilarityResult, rerank_llm, rewrite_query
from jet.file.utils import load_file, save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import LLMModelKey
from jet.llm.mlx.token_utils import chunk_text, get_tokenizer_fn, merge_texts
from jet.llm.mlx.utils.base import get_model_max_tokens
from jet.logger import logger
from jet.scrapers.utils import MergedTextsResult, merge_texts_by_hierarchy

MODEL: LLMModelKey = "llama-3.2-3b-instruct-4bit"

SYSTEM_PROMPT = """
You are a code generation assistant. Given relevant chunking logic in context, generate a Python class named Chunker. Focus on clean, efficient, and well-documented code. Output only the python code block (```python) without any other additional text.
"""


PROMPT_TEMPLATE = """
"""

mlx = MLX(MODEL)


class GenerationResult(TypedDict):
    context: str
    response: str


def call_llm_generation(context: str) -> GenerationResult:
    response = ""
    for chunk in mlx.stream_chat(
        context,
        max_tokens=-1,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.3,
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return {
        "context": context,
        "response": response,
    }


def preprocess_data(texts: list[str], chunk_size: int) -> list[str]:
    chunks = merge_texts_by_hierarchy(
        "\n\n".join(texts), get_tokenizer_fn(MODEL), max_tokens=chunk_size)
    # chunks = chunk_text(texts, chunk_size, overlap=40, model=MODEL)
    return [chunk["text"] for chunk in chunks]


class GeneratorGenerationResult(TypedDict):
    source: Union[str, Path]
    code: str
    data: list[str]
    reranked_data: list[SimilarityResult]


def run_generate_chunker_class(query: str, json_file_or_dir: str) -> Generator[GeneratorGenerationResult, None, None]:
    chunk_size = 300
    # Process the markdown file
    if os.path.isdir(json_file_or_dir):
        for file in Path(json_file_or_dir).glob('*.json'):
            logger.info(f"\nProcessing {file}")
            data: list[ProcessedResult] = load_file(str(file))
            texts = []
            for d in data:
                if d["content"].strip():
                    texts.append(f"{d['header']}\n{d["content"]}")
                if d["code"]:
                    texts.append(
                        f"{d['header']}\nCode:\n{d["code"]['content']}")
            # texts = [d["text"] for d in data]
            reranked_data = rerank_llm(query, texts)
            reranked_texts = [doc["text"] for doc in reranked_data]
            preprocessed_data = preprocess_data(reranked_texts, chunk_size)
            contexts = mlx.filter_docs(
                preprocessed_data, chunk_size=chunk_size)
            context = "\n\n".join(contexts)
            result = call_llm_generation(context)
            yield {"source": file, "code": result["response"], "data": preprocessed_data, "reranked_data": reranked_data}
    else:
        logger.info(f"\nProcessing {json_file_or_dir}")
        data: list[ProcessedResult] = load_file(str(file))
        texts = []
        for d in data:
            if d["content"].strip():
                texts.append(f"{d['header']}\n{d["content"]}")
            if d["code"]:
                texts.append(f"{d['header']}\nCode:\n{d["code"]['content']}")
        # texts = [d["text"] for d in data]
        reranked_data = rerank_llm(query, texts)
        reranked_texts = [doc["text"] for doc in reranked_data]
        preprocessed_data = preprocess_data(reranked_texts, chunk_size)
        contexts = mlx.filter_docs(preprocessed_data, chunk_size=chunk_size)
        context = "\n\n".join(contexts)
        result = call_llm_generation(context)
        yield {"source": json_file_or_dir, "code": result["response"], "data": preprocessed_data, "reranked_data": reranked_data}


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0], "markdown_processing")

    md_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/all-rag-techniques/docs/5_contextual_chunk_headers_rag.md"

    preprocess_notebooks_to_markdowns(md_dir, output_dir)

    query = "Chunking data"
    # query = rewrite_query(query)

    results = run_generate_chunker_class(query, output_dir)

    chunker_classes_output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0], "chunker_classes")

    for result in results:
        file_no_ext = os.path.splitext(os.path.basename(result['source']))[0]
        sub_dir = f"{chunker_classes_output_dir}/{file_no_ext}"
        logger.newline()
        save_file(result["code"], f"{sub_dir}/chunker.py")
        save_file(result["data"], f"{sub_dir}/data.json")
        save_file(result["reranked_data"], f"{sub_dir}/reranked_data.json")

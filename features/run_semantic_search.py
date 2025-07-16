import os
import shutil
from typing import Optional, TypedDict, List
import numpy as np
from jet.features.semantic_search import vector_search
from jet.file.utils import load_file, save_file
from jet.models.embeddings.base import generate_embeddings
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from shared.data_types.job import JobData

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"

    data: list[JobData] = load_file(data_file)
    embed_model: EmbedModelType = "all-MiniLM-L12-v2"
    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    chunk_size = 150
    query = "React web"
    top_k = None
    system = None

    texts = [
        "\n\n".join([
            f"## Job Title\n\n{item['title']}",
            f"## Details\n\n{item['details']}",
            *[
                f"## {key.replace('_', ' ').title()}\n\n" +
                "\n".join([f"- {value}" for value in item["entities"][key]])
                for key in item["entities"]
            ],
            f"## Tags\n\n" + "\n".join([f"- {tag}" for tag in item["tags"]]),
        ])
        for item in data
    ]
    save_file(texts, f"{OUTPUT_DIR}/docs.json")

    tokenizer = get_tokenizer_fn(embed_model)
    chunks = [chunk for text in texts for chunk in chunk_headers_by_hierarchy(
        text, chunk_size, tokenizer)]
    save_file(chunks, f"{OUTPUT_DIR}/chunks.json")

    texts_to_search = [
        "\n".join([
            chunk["header"],
            chunk["content"]
        ])
        for chunk in chunks
    ]

    search_results = vector_search(
        query, texts_to_search, embed_model, top_k=top_k)

    save_file({
        "query": query,
        "count": len(search_results),
        "results": search_results
    }, f"{OUTPUT_DIR}/search_results.json")

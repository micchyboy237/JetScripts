import argparse
import os
from typing import List
from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.data.utils import generate_unique_id
from jet.file.utils import save_file
from jet.logger.config import colorize_log
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.models.tokenizer.base import get_tokenizer_fn
from jet.utils.language import detect_lang
from jet.utils.text import format_sub_dir
from jet.vectors.reranker.bm25 import rerank_bm25
from jet.vectors.semantic_search.file_vector_search import FileSearchResult, merge_results, search_files


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


if __name__ == "__main__":

    query = "quality assurance agent"
    directories = [
        "/Users/jethroestrada/Desktop/External_Projects/AI",
    ]

    output_dir = f"{OUTPUT_DIR}/{format_sub_dir(query)}"

    extensions = [".py"]

    # embed_model: EmbedModelType = "jina-embeddings-v2-small"
    # sf_model = SentenceTransformerRegistry.load_model(embed_model)
    # # control your input sequence length up to 8192
    # sf_model.max_seq_length = 1024

    embed_model: EmbedModelType = "static-retrieval-mrl-en-v1"

    top_k = None
    threshold = 0.0  # Using default threshold
    chunk_size = 1000
    chunk_overlap = 100
    tokenizer = SentenceTransformerRegistry.get_tokenizer(embed_model)
    split_chunks = True

    def count_tokens(text):
        return len(tokenizer.encode(text))

    def preprocess_text(text):
        return clean_markdown_links(text)

    results = list(
        search_files(
            directories,
            query,
            extensions,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_chunks=split_chunks,
            tokenizer=count_tokens,
            preprocess=preprocess_text,
            excludes=["**/.venv/*", "**/.pytest_cache/*", "**/node_modules/*"],
            weights={
                "dir": 0.325,
                "name": 0.325,
                "content": 0.35,
            }
        )
    )
    filtered_results = [
        result for result in results
        if detect_lang(result["text"])["lang"] == "en"
    ]
    save_file({
        "query": query,
        "count": len(filtered_results),
        "merged": not split_chunks,
        "results": filtered_results
    }, f"{output_dir}/search_results.json")

import os
import shutil
from typing import List, Dict, Tuple
from jet.data.sample_diverse_texts import sample_diverse_texts
from jet.llm.mlx.generation import chat
from jet.models.model_types import LLMModelType
from jet.llm.utils.bm25_plus import bm25_plus
from jet.llm.utils.search_docs import search_docs
from jet.logger import logger
from jet.models.tokenizer.base import count_tokens
from jet.transformers.formatters import format_json
from tqdm import tqdm
import numpy as np
from jet.utils.url_utils import clean_url, parse_url, rerank_urls_bm25_plus
from jet.features.nlp_utils import get_word_counts_lemmatized
from jet.file.utils import load_file, save_file
from urllib.parse import urlparse, urlunparse
import re

from jet.vectors.document_types import HeaderDocument
from jet.wordnet.text_chunker import chunk_headers

PROMPT_TEMPLATE = """\
Context information is below.
---------------------
{context}
---------------------

Given the context information, answer the query.

Query: {query}
"""

if __name__ == "__main__":
    original_chunks_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/original_docs.json"
    merged_chunks_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/merged_chunks.json"
    rag_results_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_2/rag_results.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    original_chunks = load_file(original_chunks_file)
    merged_chunks = load_file(merged_chunks_file)
    rag_results = load_file(rag_results_file)

    original_chunks_dict = {chunk["doc_id"]
        : chunk for chunk in original_chunks}
    rag_result_merged_chunk_ids = [result["merged_chunk_id"]
                                   for result in rag_results["results"]]
    result_original_chunks = [
        original_chunks_dict[original_chunk_id]
        for merged_chunk in merged_chunks
        for original_chunk_id in merged_chunk["original_doc_ids"]
    ]
    save_file(result_original_chunks, f"{output_dir}/docs.json")
    texts = [chunk["content"] for chunk in result_original_chunks]
    save_file(texts, f"{output_dir}/doc_texts.json")

    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    query = "Top isekai anime 2025."

    diverse_result_texts: List[HeaderDocument] = sample_diverse_texts(texts)
    logger.success(f"Diverse Results: {len(diverse_result_texts)}")

    context_tokens: list[int] = count_tokens(
        llm_model, diverse_result_texts, prevent_total=True)
    total_tokens = sum(context_tokens)

    save_file({"query": query, "count": len(diverse_result_texts), "tokens": total_tokens,
              "contexts": [{"tokens": tokens, "content": result} for tokens, result in zip(context_tokens, diverse_result_texts)]}, f"{output_dir}/contexts.json")

    context = "\n\n".join(diverse_result_texts)
    save_file(context, f"{output_dir}/context.md")

    word_counts_lemmatized = get_word_counts_lemmatized(
        context, min_count=2, as_score=False)
    save_file(word_counts_lemmatized,
              f"{output_dir}/word_counts_lemmatized.json")

    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    llm_response = chat(prompt, llm_model, temperature=0.7, verbose=True)

    save_file(llm_response["content"], f"{output_dir}/response.md")

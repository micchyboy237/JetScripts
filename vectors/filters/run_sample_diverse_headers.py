import os
import shutil
from typing import List, Dict, Tuple
from jet.data.sample_diverse_headers import sample_diverse_headers
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


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    llm_model: LLMModelType = "qwen3-1.7b-4bit-dwq-053125"
    docs = load_file(docs_file)
    query = docs["query"]
    docs = docs["documents"]
    docs = [HeaderDocument(**doc) for doc in docs]
    chunked_docs = chunk_headers(docs, max_tokens=200, model=llm_model)
    docs = chunked_docs

    results: List[HeaderDocument] = sample_diverse_headers(docs)
    logger.success(f"Diverse URLs: {len(results)}")

    result_texts = [result.text for result in results]
    context_tokens: list[int] = count_tokens(
        llm_model, result_texts, prevent_total=True)
    total_tokens = sum(context_tokens)

    save_file(
        {
            "query": query,
            "total_tokens": total_tokens,
            "count": len(results),
            "urls_info": {
                result["metadata"]["source_url"]: len(
                    [r for r in results if r["metadata"]["source_url"] == result["metadata"]["source_url"]])
                for result in results
            },
            "diverse_docs": [
                {
                    "doc_index": result["doc_index"],
                    "chunk_index": result["chunk_index"],
                    "tokens": tokens,
                    "source_url": result["metadata"]["source_url"],
                    "parent_header": result["metadata"]["parent_header"],
                    "header": result["metadata"]["header"],
                    "text": result["text"]
                }
                for result, tokens in zip(results, context_tokens)
            ]
        },
        f"{output_dir}/results.json"
    )

import os
import shutil
from typing import List, Dict, Tuple
from jet.data.sample_diverse_urls import sample_diverse_urls
from jet.llm.utils.bm25_plus import bm25_plus
from jet.llm.utils.search_docs import search_docs
from jet.logger import logger
from jet.transformers.formatters import format_json
from tqdm import tqdm
import numpy as np
from jet.utils.url_utils import clean_url, parse_url, rerank_bm25_plus
from jet.features.nltk_utils import get_word_counts_lemmatized
from jet.file.utils import load_file, save_file
from urllib.parse import urlparse, urlunparse
import re


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/links.json"
    query = "List all ongoing and upcoming isekai anime 2025."
    top_k = 10

    urls: List[str] = load_file(docs_file)

    reranked_urls = rerank_bm25_plus(urls, query, top_k)
    save_file(reranked_urls, f"{output_dir}/reranked_urls.json")
    print(f"Reranked URLs: {len(reranked_urls)}")
    logger.success(format_json(reranked_urls))

    diverse_urls: List[str] = sample_diverse_urls(reranked_urls)
    save_file(diverse_urls, f"{output_dir}/diverse_urls.json")
    print(f"Diverse URLs: {len(diverse_urls)}")
    logger.success(format_json(diverse_urls))

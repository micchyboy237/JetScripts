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
from jet.utils.url_utils import clean_url, parse_url
from jet.features.nltk_utils import get_word_counts_lemmatized
from jet.file.utils import load_file, save_file
from urllib.parse import urlparse, urlunparse
import re


def preprocess_urls(urls: List[str]) -> Tuple[List[str], Dict[int, str]]:
    """Preprocess URLs into tokenized strings and maintain mapping to original URLs.

    Args:
        urls: List of URLs to preprocess.

    Returns:
        Tuple containing:
        - List of unique tokenized URLs.
        - Dictionary mapping indices of unique tokenized URLs to original URLs.
    """
    unwanted_patterns = r'wp-json|oembed|feed|xmlrpc|wp-content|wp-includes|wp-admin'
    resource_extensions = r'\.(jpg|jpeg|png|gif|bmp|pdf|zip|tar|gz|rar|css|js|woff|woff2|ttf|otf|ico|svg|mp4|mp3|avi|mov|wmv|flv|doc|docx|xls|xlsx|ppt|pptx)$'
    combined_pattern = f'({unwanted_patterns})|({resource_extensions})'
    resource_regex = re.compile(combined_pattern, re.IGNORECASE)

    tokenized_urls = []
    index_to_original_url = {}
    original_index = 0

    for url in tqdm(urls, desc="Preprocessing and filtering URLs"):
        try:
            cleaned = clean_url(url)
            if not cleaned:
                original_index += 1
                continue
            if resource_regex.search(cleaned):
                original_index += 1
                continue
            parsed = urlparse(cleaned)
            unparsed_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path,
                                      parsed.params, '', ''))  # remove query and fragment
            tokenized = ' '.join(parse_url(unparsed_url))
            tokenized_urls.append(tokenized)
            index_to_original_url[len(tokenized_urls) - 1] = url
            original_index += 1
        except ValueError as e:
            print(f"Error processing URL {url}: {e}")
            original_index += 1
            continue

    # Filter unique tokenized URLs while preserving mapping
    unique_tokenized_urls = []
    unique_index_to_original_url = {}
    seen = set()
    for idx, tokenized in enumerate(tokenized_urls):
        if tokenized not in seen:
            seen.add(tokenized)
            new_index = len(unique_tokenized_urls)
            unique_tokenized_urls.append(tokenized)
            unique_index_to_original_url[new_index] = index_to_original_url[idx]

    print(f"Retained {len(unique_tokenized_urls)} URLs after filtering")
    return unique_tokenized_urls, unique_index_to_original_url


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

    preprocessed_urls, unique_index_to_original_url = preprocess_urls(urls)
    save_file(preprocessed_urls, f"{output_dir}/preprocessed_urls.json")
    save_file(unique_index_to_original_url,
              f"{output_dir}/index_to_original_url.json")

    bm25_plus_results = bm25_plus(preprocessed_urls, query, k1=1.5)
    save_file({
        "query": query,
        **bm25_plus_results
    }, f"{output_dir}/bm25_plus_results.json")

    # Map doc_index to original URLs and debug
    reranked_urls = []
    for result in bm25_plus_results["results"]:
        doc_index = result["doc_index"]
        score = result["score"]
        if score > 0.9 and doc_index in unique_index_to_original_url:
            original_url = unique_index_to_original_url[doc_index]
            reranked_urls.append(original_url)

    # Unique results and limit to top_k
    reranked_urls = list(dict.fromkeys(reranked_urls))
    reranked_urls = reranked_urls[:top_k]
    save_file(reranked_urls, f"{output_dir}/reranked_urls.json")
    print(f"Reranked URLs: {len(reranked_urls)}")
    logger.success(format_json(reranked_urls))

    diverse_urls: List[str] = sample_diverse_urls(reranked_urls)
    save_file(diverse_urls, f"{output_dir}/diverse_urls.json")
    print(f"Diverse URLs: {len(diverse_urls)}")
    logger.success(format_json(diverse_urls))

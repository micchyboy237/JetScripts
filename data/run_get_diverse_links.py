import os
import shutil
from typing import List
from urllib.parse import urlparse, parse_qs

from jet.data.stratified_sampler import ProcessedDataString, StratifiedSampler
from jet.data.url_sampler import preprocess_url, sample_diverse_urls
from jet.features.nltk_utils import get_word_counts_lemmatized
from jet.file.utils import load_file, save_file
from jet.wordnet.similarity import filter_similar_texts


def preprocess_urls(urls: List[str]):
    return [preprocess_url(url) for url in urls]


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/links.json"

    # Load JSON data
    urls: List[str] = load_file(docs_file)

    # Get word counts
    word_counts_lemmatized_results = get_word_counts_lemmatized(
        urls, pos=["noun"], min_count=2, with_score=True)
    save_file(word_counts_lemmatized_results,
              f"{output_dir}/word-counts-lemmatized-results.json")

    # Filter out similar urls
    diverse_urls = filter_similar_texts(urls, show_progress=True)

    save_file(diverse_urls, f"{output_dir}/diverse-urls.json")

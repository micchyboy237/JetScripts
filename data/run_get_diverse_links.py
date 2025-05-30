import os
import shutil
from typing import List
from urllib.parse import urlparse, parse_qs

from jet.data.url_sampler import preprocess_url, sample_diverse_urls
from jet.file.utils import load_file, save_file


def preprocess_urls(urls: List[str]):
    return [preprocess_url(url) for url in urls]


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/links.json"

    # Load JSON data
    links: List[str] = load_file(docs_file)

    # Preprocess links
    urls = preprocess_urls(links)
    save_file(links, f"{output_dir}/preprocessed-urls.json")

    # Stratify links for diversity
    num_samples = 1000
    n = 2
    top_n = 2
    diverse_urls = sample_diverse_urls(links, num_samples, n, top_n)
    save_file(diverse_urls, f"{output_dir}/diverse-urls.json")

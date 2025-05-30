import os
import shutil
from typing import List
from urllib.parse import urlparse, parse_qs

from jet.data.stratified_sampler import ProcessedDataString, StratifiedSampler
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
    urls: List[str] = load_file(docs_file)

    # Sample 1000 diverse URLs
    num_samples = 1000
    n = 2
    top_n = 2
    sampled_urls = sample_diverse_urls(urls, num_samples, n, top_n)
    save_file(sampled_urls, f"{output_dir}/sampled-urls.json")

    # For Scenario 2: Create train/test/val datasets
    # Assign categories (e.g., based on domain)
    data = [
        ProcessedDataString(
            source=url,
            category_values=[urlparse(url).netloc]
        )
        for url in sampled_urls
    ]

    # Split into train (60%), test (20%), val (20%)
    sampler = StratifiedSampler(data)
    train_data, test_data, val_data = sampler.split_train_test_val(
        train_ratio=0.6, test_ratio=0.2)

    print(
        f"Train: {len(train_data)}, Test: {len(test_data)}, Val: {len(val_data)}")
    save_file(train_data, f"{output_dir}/train_data.json")
    save_file(test_data, f"{output_dir}/test_data.json")
    save_file(val_data, f"{output_dir}/val_data.json")

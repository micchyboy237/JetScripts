import os
import shutil
from typing import List, Dict, Tuple
from jet.llm.utils.search_docs import search_docs
from tqdm import tqdm
import numpy as np
from jet.utils.url_utils import clean_url, parse_url
from jet.features.nltk_utils import get_word_counts_lemmatized
from jet.file.utils import load_file, save_file
from urllib.parse import urlparse, urlunparse
import re  # Added for pattern matching


def preprocess_urls(urls: List[str]) -> List[str]:
    """Preprocess URLs into tokenized strings, removing hash fragments and filtering resources."""
    # Define patterns for resource-related URLs
    unwanted_patterns = r'wp-json|oembed|feed|xmlrpc|wp-content|wp-includes|wp-admin'
    resource_extensions = r'\.(jpg|jpeg|png|gif|bmp|pdf|zip|tar|gz|rar|css|js|woff|woff2|ttf|otf|ico|svg|mp4|mp3|avi|mov|wmv|flv|doc|docx|xls|xlsx|ppt|pptx)$'
    combined_pattern = f'({unwanted_patterns})|({resource_extensions})'
    resource_regex = re.compile(combined_pattern, re.IGNORECASE)

    tokenized_urls = []
    for url in tqdm(urls, desc="Preprocessing and filtering URLs"):
        try:
            # Clean URL first
            cleaned = clean_url(url)
            if not cleaned:
                continue
            # Filter out URLs matching resource patterns or extensions
            if resource_regex.search(cleaned):
                print(f"Filtered out resource URL: {cleaned}")
                continue
            # Parse URL and remove fragment
            parsed = urlparse(cleaned)
            unparsed_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path,
                                      parsed.params, parsed.query, ''))
            # Preprocess the cleaned URL
            tokenized = ' '.join(parse_url(unparsed_url))
            tokenized_urls.append(tokenized)
        except ValueError as e:
            print(f"Error processing URL {url}: {e}")
            continue
    print(f"Retained {len(tokenized_urls)} URLs after resource filtering")
    return tokenized_urls


def postprocess_urls(urls: List[str]) -> List[str]:
    """Clean URLs."""
    cleaned_urls = []
    for url in tqdm(urls, desc="Cleaning URLs"):
        try:
            cleaned = clean_url(url)
            if cleaned:
                cleaned_urls.append(cleaned)
        except ValueError as e:
            print(f"Error cleaning URL {url}: {e}")
            continue
    print(f"Retained {len(cleaned_urls)} URLs after cleaning")
    return cleaned_urls


def get_query_noun_profile(query: str) -> Dict[str, float]:
    """Extract lemmatized nouns and their scores from the query."""
    return get_word_counts_lemmatized(
        query, pos=None, min_count=1, as_score=True
    )


def map_urls_to_nouns(
    urls: List[str], word_counts: Dict[str, float], query_profile: Dict[str, float]
) -> List[Tuple[Dict[str, float], float]]:
    """Map each URL to its lemmatized nouns and compute relevance to query."""
    preprocessed_urls = preprocess_urls(urls)
    noun_profiles = []
    all_nouns = set(word_counts.keys())

    for preprocessed in tqdm(preprocessed_urls, desc="Mapping URLs to nouns"):
        url_counts = get_word_counts_lemmatized(
            preprocessed, pos=None, min_count=1, as_score=True
        )
        noun_profile = {noun: score for noun,
                        score in url_counts.items() if noun in all_nouns}
        relevance = cosine_similarity(
            noun_profile, query_profile) if noun_profile and query_profile else 0.0
        noun_profiles.append((noun_profile, relevance))

    return noun_profiles


def cosine_similarity(profile1: Dict[str, float], profile2: Dict[str, float]) -> float:
    """Compute cosine similarity between two noun profiles."""
    all_nouns = set(profile1.keys()) | set(profile2.keys())
    if not all_nouns:
        return 0.0
    vec1 = np.array([profile1.get(noun, 0.0) for noun in all_nouns])
    vec2 = np.array([profile2.get(noun, 0.0) for noun in all_nouns])
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def filter_diverse_urls_by_nouns(
    urls: List[str],
    noun_profiles: List[Tuple[Dict[str, float], float]],
    relevance_threshold: float = 0.3,
    diversity_threshold: float = 0.7,
    relevance_weight: float = 0.5,
    show_progress: bool = False
) -> List[str]:
    """Filter URLs to retain diverse and relevant ones based on noun profiles."""
    filtered_urls = []
    filtered_profiles = []
    iterator = tqdm(zip(urls, noun_profiles),
                    desc="Filtering diverse URLs") if show_progress else zip(urls, noun_profiles)

    sorted_pairs = sorted(
        zip(urls, noun_profiles),
        key=lambda x: x[1][1],
        reverse=True
    )

    for url, (profile, relevance) in iterator:
        if not profile or relevance < relevance_threshold:
            continue
        is_diverse = True
        for existing_profile, _ in filtered_profiles:
            if cosine_similarity(profile, existing_profile) >= diversity_threshold:
                is_diverse = False
                break
        if is_diverse:
            filtered_urls.append(url)
            filtered_profiles.append((profile, relevance))

    print(f"Retained {len(filtered_urls)} diverse and relevant URLs")
    return filtered_urls


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/links.json"
    query = "List all ongoing and upcoming isekai anime 2025."
    urls: List[str] = load_file(docs_file)

    preprocessed_urls = preprocess_urls(urls)
    save_file(preprocessed_urls, f"{output_dir}/preprocessed-urls.json")

    word_counts_lemmatized_results = get_word_counts_lemmatized(
        '\n\n'.join(preprocessed_urls), pos=["noun"], min_count=2, as_score=True
    )
    save_file(word_counts_lemmatized_results,
              f"{output_dir}/word-counts-lemmatized-results.json")

    query_profile = get_query_noun_profile(query)

    noun_profiles = map_urls_to_nouns(
        urls, word_counts_lemmatized_results, query_profile)
    save_file(noun_profiles, f"{output_dir}/noun-profiles.json")

    diverse_urls = filter_diverse_urls_by_nouns(
        urls, noun_profiles, relevance_threshold=0.3, diversity_threshold=0.7, show_progress=True
    )
    diverse_urls = postprocess_urls(diverse_urls)
    save_file(diverse_urls, f"{output_dir}/diverse-urls.json")

    searched_diverse_urls = search_docs(
        query, diverse_urls, model="snowflake-arctic-embed:33m")
    save_file(searched_diverse_urls,
              f"{output_dir}/searched-diverse-urls.json")

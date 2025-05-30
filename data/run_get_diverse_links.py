import os
import shutil
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
from jet.data.url_sampler import preprocess_url
from jet.features.nltk_utils import get_word_counts_lemmatized
from jet.file.utils import load_file, save_file


def preprocess_urls(urls: List[str]) -> List[str]:
    """Preprocess URLs into tokenized strings."""
    return [' '.join(preprocess_url(url)) for url in tqdm(urls, desc="Preprocessing URLs")]


def get_query_noun_profile(query: str) -> Dict[str, float]:
    """Extract lemmatized nouns and their scores from the query."""
    return get_word_counts_lemmatized(
        query, pos=["noun"], min_count=1, as_score=True
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
        # Filter to only include nouns present in the global word_counts
        noun_profile = {noun: score for noun,
                        score in url_counts.items() if noun in all_nouns}
        # Compute relevance score (cosine similarity to query profile)
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

    # Sort URLs by relevance score to prioritize relevant ones
    sorted_pairs = sorted(
        zip(urls, noun_profiles),
        key=lambda x: x[1][1],  # Sort by relevance score
        reverse=True
    )

    for url, (profile, relevance) in iterator:
        if not profile or relevance < relevance_threshold:  # Skip irrelevant URLs
            continue
        # Check if the noun profile is too similar to any already selected
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

    # Preprocess URLs
    preprocessed_urls = preprocess_urls(urls)
    save_file(preprocessed_urls, f"{output_dir}/preprocessed-urls.json")

    # Get lemmatized word counts for all URLs
    word_counts_lemmatized_results = get_word_counts_lemmatized(
        '\n\n'.join(preprocessed_urls), pos=["noun"], min_count=2, as_score=True
    )
    save_file(word_counts_lemmatized_results,
              f"{output_dir}/word-counts-lemmatized-results.json")

    # Get query noun profile
    query_profile = get_query_noun_profile(query)

    # Map URLs to their noun profiles and relevance scores
    noun_profiles = map_urls_to_nouns(
        urls, word_counts_lemmatized_results, query_profile)
    save_file(noun_profiles, f"{output_dir}/noun-profiles.json")

    # Filter diverse and relevant URLs
    diverse_urls = filter_diverse_urls_by_nouns(
        urls, noun_profiles, relevance_threshold=0.3, diversity_threshold=0.7, show_progress=True
    )
    save_file(diverse_urls, f"{output_dir}/diverse-urls.json")

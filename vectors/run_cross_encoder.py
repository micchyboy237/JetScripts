import os
from typing import List, Dict, Tuple, Union
from typing import Literal
import numpy as np
import logging
from jet.code.markdown_utils._markdown_parser import parse_markdown
from jet.data.header_utils._prepare_for_rag import preprocess_text
from jet.file.utils import load_file, save_file
from jet.models.model_registry.transformers.cross_encoder_model_registry import CrossEncoderRegistry

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration dictionary for modularity
CONFIG = {
    "ngrams_file": "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/wordnet/n_grams/generated/run_ngram_examples/filtered_ngrams_min_count_2.json",
    "queries": [
        "Tell me about your recent achievements.",
        "What are the latest trends in artificial intelligence?",
        "How can I improve my software development skills?",
        "Looking for insights on cloud computing advancements.",
        "Give me tips for effective project management."
    ],
    "output_dir": os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    ),
    "normalization_method": "minmax"  # Options: "sigmoid", "minmax"
}


def normalize_scores(scores: Union[List[float], np.ndarray], method: Literal["sigmoid", "minmax"] = "sigmoid") -> List[float]:
    """Normalize raw cross-encoder scores using specified method.

    Args:
        scores: List or NumPy array of raw logits from cross-encoder model.
        method: Normalization method ('sigmoid' or 'minmax').

    Returns:
        List of normalized scores in [0, 1].

    Raises:
        ValueError: If method is unsupported or scores are invalid for minmax.
        ValueError: If scores contain NaN or infinite values.
    """
    # Convert NumPy array to list if necessary
    if isinstance(scores, np.ndarray):
        scores = scores.tolist()

    if len(scores) == 0:
        logger.warning("Empty scores list provided, returning empty list")
        return []

    # Check for invalid values
    if any(not np.isfinite(score) for score in scores):
        raise ValueError("Scores contain NaN or infinite values")

    if method == "sigmoid":
        logger.debug("Applying sigmoid normalization")
        return [1 / (1 + np.exp(-score)) for score in scores]
    elif method == "minmax":
        logger.debug("Applying min-max normalization")
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            logger.warning(
                "Max and min scores are equal, returning 0.5 for all scores")
            return [0.5] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def main() -> None:
    """Main function to demonstrate semantic search with real-world queries."""
    # Ensure output directory exists
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    resume_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-resume/complete_jet_resume.md"
    markdown_tokens = parse_markdown(
        resume_path, merge_headers=False, merge_contents=True, ignore_links=False)
    docs = [d["content"] for d in markdown_tokens]

    # Load n-grams dictionary
    logger.info(f"Loading n-grams from {CONFIG['ngrams_file']}")
    # ngrams_dict: Dict[str, int] = load_file(CONFIG["ngrams_file"])
    # docs: List[str] = list(ngrams_dict.keys())
    logger.info(f"Loaded {len(docs)} documents")

    # Load model
    logger.info("Loading cross-encoder model")
    model = CrossEncoderRegistry.load_model('ms-marco-MiniLM-L6-v2')

    # Create pairs of each query with each document
    pairs: List[Tuple[str, str]] = [(query, doc)
                                    for query in CONFIG["queries"] for doc in docs]
    logger.info(f"Created {len(pairs)} query-document pairs")

    # Compute relevance scores
    logger.info("Computing relevance scores")
    raw_scores = model.predict(pairs)

    # Normalize scores
    try:
        scores = normalize_scores(
            raw_scores, method=CONFIG["normalization_method"])
    except ValueError as e:
        logger.error(f"Normalization failed: {str(e)}")
        raise

    # Create ranked results with query information
    ranked_results = []
    for i, (query, doc) in enumerate(pairs):
        ranked_results.append((query, doc, scores[i]))

    # Sort by score in descending order
    ranked_results.sort(key=lambda x: x[2], reverse=True)

    # Log top 10 results
    logger.info("Top 10 ranked results:")
    for query, doc, score in ranked_results[:10]:
        logger.info(f"Query: {query}, Doc: {doc}, Score: {score}")

    # Save results
    logger.info(f"Saving scores to {CONFIG['output_dir']}/scores.json")
    save_file(scores, f"{CONFIG['output_dir']}/scores.json")
    logger.info(
        f"Saving ranked results to {CONFIG['output_dir']}/ranked_results.json")
    save_file(ranked_results, f"{CONFIG['output_dir']}/ranked_results.json")


if __name__ == "__main__":
    main()

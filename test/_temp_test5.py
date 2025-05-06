import os
from typing import List, Dict
from jet.file.utils import load_file
from jet.llm.mlx.templates.generate_labels import generate_labels
from jet.logger.logger import CustomLogger

# Initialize logger
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


def preprocess_text(text: str) -> str:
    """Normalize and clean text for NER."""
    if not isinstance(text, str):
        return ""
    return text.strip().lower()


def load_and_validate_files(site_info_file: str, queries_file: str) -> tuple[Dict, Dict]:
    """Load and validate input files."""
    try:
        site_info = load_file(site_info_file)
        queries = load_file(queries_file)
        if not isinstance(site_info, dict) or not isinstance(queries, dict):
            raise ValueError("Invalid file content: Expected dictionaries")
        return site_info, queries
    except Exception as e:
        logger.error(f"Failed to load files: {e}")
        raise


def prepare_text(site_info: Dict, queries: Dict) -> List[str]:
    """Prepare text inputs for label generation."""
    query = queries.get("query", "")
    title = site_info.get("title", "")
    metadata = site_info.get("metadata", {})
    description = metadata.get("description", "")
    keywords = metadata.get("keywords", "")

    # Preprocess individual fields
    text = [preprocess_text(title), preprocess_text(description)]

    # Split and preprocess keywords
    if isinstance(keywords, str):
        keyword_list = [preprocess_text(kw)
                        for kw in keywords.split(",") if kw.strip()]
        text.extend(keyword_list)
    elif isinstance(keywords, list):
        text.extend(preprocess_text(kw) for kw in keywords if kw.strip())

    text.append(preprocess_text(query))

    # Remove duplicates while preserving order
    text = list(dict.fromkeys(t for t in text if t))

    return text


def main():
    # File paths
    headers_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_html_myanimelist_net_Isekai/headers.json"
    site_info_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/searched_html_myanimelist_net_Isekai/context_info.json"
    queries_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/queries.json"

    try:
        # Load files
        site_info, queries = load_and_validate_files(
            site_info_file, queries_file)

        # Prepare text inputs
        text = prepare_text(site_info, queries)
        logger.info(f"Preprocessed text: {text}")

        # Generate labels
        results = generate_labels(text)
        if results:
            logger.info(f"Generated labels: {results}")
        else:
            logger.error("No labels generated.")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()

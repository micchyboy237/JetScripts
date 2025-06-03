import os
import shutil
from typing import List, Dict, Union, Any
from jet.features.nlp_utils import get_word_counts_lemmatized, get_word_sentence_combination_counts, get_document_summary
from jet.features.nlp_utils.nlp_types import WordOccurrence, Matched
from jet.file.utils import load_file, save_file


def setup_output_directory(script_file: str) -> str:
    """Create and return the output directory path for saving results."""
    output_dir = os.path.join(
        os.path.dirname(script_file), "generated", os.path.splitext(
            os.path.basename(script_file))[0]
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_documents(file_path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSON file and return them."""
    docs = load_file(file_path)
    print(f"Loaded JSON data {len(docs)} from: {file_path}")
    return docs


def extract_texts(docs: List[Dict[str, Any]]) -> tuple[List[str], str]:
    """Extract texts from documents and return both list and joined string."""
    docs_text = [doc["text"] for doc in docs]
    docs_str = "\n\n".join(docs_text)
    return docs_text, docs_str


def save_results(results: Any, output_dir: str, filename: str) -> None:
    """Save results to a JSON file in the output directory."""
    output_path = os.path.join(output_dir, filename)
    save_file(results, output_path)
    print(f"Save JSON data to: {output_path}")


def process_word_counts_lemmatized(text: Union[str, List[str]], output_dir: str, min_count: int = 2) -> None:
    """Process lemmatized word counts for text(s) and save results."""
    # Process with raw counts
    results_counts = get_word_counts_lemmatized(
        text, min_count=min_count, as_score=False)
    save_results(
        results_counts,
        output_dir,
        f"word_counts_lemmatized_{'list_' if isinstance(text, list) else ''}counts.json"
    )

    # Process with normalized scores
    results_scores = get_word_counts_lemmatized(
        text, min_count=min_count, as_score=True)
    save_results(
        results_scores,
        output_dir,
        f"word_counts_lemmatized_{'list_' if isinstance(text, list) else ''}scores.json"
    )


def process_word_sentence_combination_counts(
    text: Union[str, List[str]], output_dir: str, n: int = 1, min_count: int = 5, in_sequence: bool = False
) -> None:
    """Process word sentence combination counts for text(s) and save results."""
    results = get_word_sentence_combination_counts(
        text, n=n, min_count=min_count, in_sequence=in_sequence, show_progress=True
    )
    suffix = f"_n_{n}_sequence" if in_sequence and n > 1 else ""
    save_results(
        results,
        output_dir,
        f"word_sentence_combination{'_list' if isinstance(text, list) else ''}{suffix}_counts.json"
    )


def process_document_summary(texts: List[str], output_dir: str, queries: List[str]) -> None:
    """Process document summary for texts with queries and save results."""
    # Process with normalized scores
    summary_scores = get_document_summary(
        texts, queries, min_count=1, as_score=True)
    save_results(summary_scores, output_dir, "document_summary_scores.json")

    # Process with raw counts
    summary_counts = get_document_summary(
        texts, queries, min_count=1, as_score=False)
    save_results(summary_counts, output_dir, "document_summary_counts.json")


def main() -> None:
    """Main function to orchestrate NLP processing tasks."""
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = setup_output_directory(__file__)
    docs = load_documents(docs_file)
    docs_text, docs_str = extract_texts(docs)

    # Process lemmatized word counts
    process_word_counts_lemmatized(docs_str, output_dir)
    process_word_counts_lemmatized(docs_text, output_dir)

    # Process word sentence combination counts
    process_word_sentence_combination_counts(
        docs_str, output_dir, n=1, min_count=5, in_sequence=False)
    process_word_sentence_combination_counts(
        docs_text, output_dir, n=1, min_count=5, in_sequence=False)
    process_word_sentence_combination_counts(
        docs_str, output_dir, n=2, min_count=5, in_sequence=True)
    process_word_sentence_combination_counts(
        docs_text, output_dir, n=2, min_count=5, in_sequence=True)

    # Process document summary with example queries
    queries = ["quick", "lazy", "fox"]  # Example queries
    process_document_summary(docs_text, output_dir, queries)


if __name__ == "__main__":
    main()

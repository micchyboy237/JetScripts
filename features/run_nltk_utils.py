import os
from jet.features.nltk_utils import get_word_counts_lemmatized, get_word_sentence_combination_counts
from jet.file.utils import load_file, save_file


# Example usage
if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    headers: list[dict] = load_file(docs_file)
    docs = [header["text"] for header in headers]

    word_counts_lemmatized_results = get_word_counts_lemmatized(
        "\n\n".join(docs))
    save_file(word_counts_lemmatized_results,
              f"{output_dir}/word_counts_lemmatized.json")

    word_sentence_combination_counts_results = get_word_sentence_combination_counts(
        docs, n=None, min_count=2, in_sequence=False)
    save_file(word_sentence_combination_counts_results,
              f"{output_dir}/word_sentence_combination_counts.json")

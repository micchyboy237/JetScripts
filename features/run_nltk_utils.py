import os
from jet.features.nltk_utils import get_word_counts_lemmatized, get_word_sentence_combination_counts
from jet.file.utils import load_file, save_file

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"

    # Load JSON data
    headers = load_file(docs_file)
    print(f"Loaded JSON data {len(headers)} from: {docs_file}")
    docs = [header["text"] for header in headers]
    docs_str = "\n\n".join(docs)

    # Process word counts for each document separately
    word_counts_lemmatized_results = get_word_counts_lemmatized(
        docs, pos=["noun"], min_count=2)
    output_path = f"{output_dir}/word_counts_lemmatized.json"
    save_file(word_counts_lemmatized_results, output_path)
    print(f"Save JSON data to: {output_path}")

    # Process word sentence combination counts
    word_sentence_combination_counts_results = get_word_sentence_combination_counts(
        docs, n=5, min_count=2, in_sequence=False, show_progress=True)
    output_path = f"{output_dir}/word_sentence_combination_counts.json"
    save_file(word_sentence_combination_counts_results, output_path)
    print(f"Save JSON data to: {output_path}")

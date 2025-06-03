import os
import shutil
from jet.features.nltk_utils import get_word_counts_lemmatized, get_word_sentence_combination_counts
from jet.file.utils import load_file, save_file

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"

    # Load JSON data
    docs = load_file(docs_file)
    print(f"Loaded JSON data {len(docs)} from: {docs_file}")
    docs = [doc["text"] for doc in docs]
    docs_str = "\n\n".join(docs)

    # Process word counts for each document as a whole
    word_counts_lemmatized_text_results = get_word_counts_lemmatized(
        docs_str, min_count=2, as_score=False)
    output_path = f"{output_dir}/word_counts_lemmatized_counts.json"
    save_file(word_counts_lemmatized_text_results, output_path)
    print(f"Save JSON data to: {output_path}")
    word_counts_lemmatized_text_results = get_word_counts_lemmatized(
        docs_str, min_count=2, as_score=True)
    output_path = f"{output_dir}/word_counts_lemmatized_scores.json"
    save_file(word_counts_lemmatized_text_results, output_path)
    print(f"Save JSON data to: {output_path}")

    # Process word counts for each document separately
    word_counts_lemmatized_list_results = get_word_counts_lemmatized(
        docs, min_count=2, as_score=False)
    output_path = f"{output_dir}/word_counts_lemmatized_list_counts.json"
    save_file(word_counts_lemmatized_list_results, output_path)
    print(f"Save JSON data to: {output_path}")

    word_counts_lemmatized_list_results = get_word_counts_lemmatized(
        docs, min_count=2, as_score=True)
    output_path = f"{output_dir}/word_counts_lemmatized_list_scores.json"
    save_file(word_counts_lemmatized_list_results, output_path)
    print(f"Save JSON data to: {output_path}")

    # Process word sentence combination counts as a whole
    word_sentence_combination_counts_results = get_word_sentence_combination_counts(
        docs_str, n=1, min_count=5, in_sequence=False, show_progress=True)
    output_path = f"{output_dir}/word_sentence_combination_counts.json"
    save_file(word_sentence_combination_counts_results, output_path)
    print(f"Save JSON data to: {output_path}")

    # Process word sentence for each document separately
    word_sentence_combination_counts_results = get_word_sentence_combination_counts(
        docs, n=1, min_count=5, in_sequence=False, show_progress=True)
    output_path = f"{output_dir}/word_sentence_combination_list_counts.json"
    save_file(word_sentence_combination_counts_results, output_path)
    print(f"Save JSON data to: {output_path}")

    # Process word sentence combination counts as a whole
    word_sentence_combination_counts_results = get_word_sentence_combination_counts(
        docs_str, n=2, min_count=5, in_sequence=True, show_progress=True)
    output_path = f"{output_dir}/word_sentence_combination_n_2_sequence_counts.json"
    save_file(word_sentence_combination_counts_results, output_path)
    print(f"Save JSON data to: {output_path}")

    # Process word sentence for each document separately
    word_sentence_combination_counts_results = get_word_sentence_combination_counts(
        docs, n=2, min_count=5, in_sequence=True, show_progress=True)
    output_path = f"{output_dir}/word_sentence_combination_n_2_sequence_list_counts.json"
    save_file(word_sentence_combination_counts_results, output_path)
    print(f"Save JSON data to: {output_path}")

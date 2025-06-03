import os
import shutil
from jet.data.stratified_sampler import filter_and_sort_sentences_by_ngrams
from jet.features.nltk_utils import get_word_counts_lemmatized, get_word_sentence_combination_counts
from jet.file.utils import load_file, save_file
from jet.wordnet.n_grams import count_ngrams, filter_texts_by_multi_ngram_count, get_common_texts, get_most_common_ngrams, get_ngrams_by_range, group_sentences_by_ngram

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"

    # Load JSON data
    docs = load_file(docs_file)
    texts = [doc["text"] for doc in docs]

    results = count_ngrams(texts, min_count=5, max_words=10)
    save_file(results, f"{output_dir}/count_ngrams.json")

    results = count_ngrams(
        texts, min_count=5, max_words=10, case_insensitive=True)
    save_file(results, f"{output_dir}/count_ngrams_lowered.json")

    # results = get_common_texts([text.lower() for text in texts])
    # save_file(results, f"{output_dir}/get_common_texts.json")

    results = get_most_common_ngrams(texts, min_count=1, max_words=5)
    save_file(results, f"{output_dir}/get_most_common_ngrams.json")

    results = get_ngrams_by_range(texts, min_words=1, max_words=3, count=(2,))
    save_file(results, f"{output_dir}/get_ngrams_by_range.json")

    results = filter_texts_by_multi_ngram_count(
        texts, min_words=1, max_words=3, count=(2,), count_all_ngrams=True)
    save_file(results, f"{output_dir}/filter_texts_by_multi_ngram_count.json")

    results = group_sentences_by_ngram(
        texts, min_words=2, top_n=2, is_start_ngrams=False)
    save_file(results, f"{output_dir}/group_sentences_by_ngram.json")

    results = filter_and_sort_sentences_by_ngrams(texts, 1, 1, True)
    save_file(
        results, f"{output_dir}/filter_and_sort_sentences_by_ngrams.json")

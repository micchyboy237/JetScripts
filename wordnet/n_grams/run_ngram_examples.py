from jet.libs.bertopic.examples.mock import load_sample_data
from jet.wordnet.keywords.helpers import preprocess_texts
import os
import shutil
from jet.file.utils import save_file
from jet.wordnet.n_grams import (
    calculate_n_gram_diversity,
    count_ngrams,
    extract_ngrams,
    count_ngrams_with_texts,
    filter_texts_by_multi_ngram_count,
    get_common_texts,
    get_most_common_ngrams,
    get_ngram_weight,
    get_ngrams,
    get_ngrams_by_range,
    get_specific_ngram_count,
    get_total_counts_of_ngrams,
    get_total_unique_ngrams,
    group_sentences_by_ngram,
    filter_and_sort_sentences_by_ngrams,
    n_gram_frequency,
    separate_ngram_lines,
    sort_sentences,
    nwise,
)

# from jet.wordnet.pos_tagger import POSTagger

from jet.logger import logger

OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    # texts = [
    #     "Similar text 1",
    #     "Similar text 2",
    #     "Different writing 3",
    #     "Different writing 4",
    # ]
    texts = load_sample_data()

    logger.debug(f"Loaded {len(texts)} documents")

    texts = preprocess_texts(texts)
    save_file(texts, f"{OUTPUT_DIR}/1_preprocessed_texts.json")

    # Existing function calls
    all_ngrams = count_ngrams([text.lower()
                               for text in texts], min_words=1)
    save_file(all_ngrams, f"{OUTPUT_DIR}/2_all_ngrams_count.json")

    filtered_ngrams = count_ngrams(
        [text.lower() for text in texts], min_count=2)
    save_file(filtered_ngrams,
              f"{OUTPUT_DIR}/3_filtered_ngrams_min_count_2.json")

    result = get_most_common_ngrams([text.lower() for text in texts])
    save_file(result, f"{OUTPUT_DIR}/4_most_common_ngrams.json")

    result = get_common_texts(
        [text.lower() for text in texts], includes_pos=["PROPN", "NOUN", "VERB", "ADJ", "ADV"],)
    save_file(result, f"{OUTPUT_DIR}/5_common_texts.json")

    result = group_sentences_by_ngram(
        [text.lower() for text in texts], is_start_ngrams=False, top_n=10)
    save_file(result, f"{OUTPUT_DIR}/6_grouped_sentences.json")

    lowered_ngrams_count = count_ngrams(
        [text.lower() for text in texts], min_words=1, max_words=3)
    save_file(lowered_ngrams_count,
              f"{OUTPUT_DIR}/7_lowered_ngrams_count.json")

    range_results = list(get_ngrams_by_range(
        texts, min_words=1, max_words=3, count=(2,), show_count=True))
    save_file(range_results, f"{OUTPUT_DIR}/8_ngrams_by_range.json")

    count_results = list(get_ngrams_by_range(
        texts, min_words=2, count=2, show_count=True))
    save_file(count_results, f"{OUTPUT_DIR}/9_ngrams_by_count.json")

    results = filter_texts_by_multi_ngram_count(
        texts, min_words=1, count=(2,), count_all_ngrams=True)
    save_file(results, f"{OUTPUT_DIR}/10_filtered_texts.json")

    # Calculate n-gram frequency
    ngram_freq = n_gram_frequency(
        " ".join([text.lower() for text in texts]), n=2)
    save_file(ngram_freq, f"{OUTPUT_DIR}/11_ngram_frequency.json")

    # Calculate n-gram diversity
    diversity = calculate_n_gram_diversity(ngram_freq)
    save_file({"diversity": diversity},
              f"{OUTPUT_DIR}/12_ngram_diversity.json")

    # Separate n-gram lines
    separated_lines = separate_ngram_lines(
        [text.lower() for text in texts], punctuations_split=[',', '/', ':'])
    save_file(separated_lines, f"{OUTPUT_DIR}/13_separated_ngram_lines.json")

    # Get n-grams
    ngrams_list = get_ngrams([text.lower()
                             for text in texts], min_words=1, max_words=2)
    save_file(ngrams_list, f"{OUTPUT_DIR}/14_ngrams_list.json")

    # Get n-gram weight for the first sentence
    sentence_ngrams = extract_ngrams(texts, min_words=1, max_words=2)
    save_file(sentence_ngrams, f"{OUTPUT_DIR}/15_extract_ngrams.json")

    previous_ngrams = set()  # Empty for example; adjust based on context
    weight = get_ngram_weight(
        all_ngrams, sentence_ngrams, previous_ngrams)
    save_file({"weight": weight}, f"{OUTPUT_DIR}/16_ngram_weight.json")

    ngram_results = count_ngrams_with_texts(
        texts=[text.lower() for text in texts],
        min_words=1,
        min_count=2,
        max_words=2
    )
    output_path = f"{OUTPUT_DIR}/17_ngrams_with_texts.json"
    save_file(ngram_results, output_path)

    # Sort sentences
    sorted_sentences = sort_sentences(
        [text.lower() for text in texts], n=2)
    save_file(sorted_sentences, f"{OUTPUT_DIR}/18_sorted_sentences.json")

    # Filter and sort sentences by n-grams
    filtered_sorted_sentences = filter_and_sort_sentences_by_ngrams(
        [text.lower() for text in texts], n=2, top_n=2, is_start_ngrams=False
    )
    save_file(filtered_sorted_sentences,
              f"{OUTPUT_DIR}/19_filtered_sorted_sentences.json")

    # Get total unique n-grams
    total_unique = get_total_unique_ngrams(all_ngrams)
    save_file({"total_unique_ngrams": total_unique},
              f"{OUTPUT_DIR}/20_total_unique_ngrams.json")

    # Get total counts of n-grams
    total_counts = get_total_counts_of_ngrams(all_ngrams)
    save_file({"total_ngram_counts": total_counts},
              f"{OUTPUT_DIR}/21_total_ngram_counts.json")

    # Get specific n-gram count (using first n-gram as example)
    specific_ngram = next(iter(all_ngrams))
    specific_count = get_specific_ngram_count(
        all_ngrams, specific_ngram)
    save_file(
        {"ngram": specific_ngram, "count": specific_count},
        f"{OUTPUT_DIR}/22_specific_ngram_count.json"
    )

    # Use nwise to generate sliding window n-grams
    nwise_ngrams = [" ".join(ngram) for ngram in nwise(texts, n=2)]
    save_file(nwise_ngrams, f"{OUTPUT_DIR}/23_nwise_ngrams.json")

from jet.code.markdown_utils._markdown_parser import parse_markdown
from jet.wordnet.lemmatizer import lemmatize_text
from jet.search.formatters import clean_string
from jet.scrapers.utils import clean_newlines, clean_punctuations, clean_spaces
import os
import shutil
from typing import List, Dict, Union
from jet.file.utils import load_file, save_file
from jet.vectors.document_types import HeaderDocument
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
import logging

from jet.wordnet.pos_tagger import POSTagger
from jet.wordnet.words import get_words

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_texts(texts: str | list[str]) -> list[str]:
    from tqdm import tqdm
    from nltk.corpus import stopwords
    import nltk

    # Download stopwords if not already downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    if isinstance(texts, str):
        texts = [texts]

    # Lowercase
    # texts = [text.lower() for text in texts]
    preprocessed_texts: list[str] = texts.copy()
    stop_words = set(stopwords.words('english'))

    tagger = POSTagger()

    for idx, text in enumerate(tqdm(preprocessed_texts, desc="Preprocessing texts")):
        # Filter words by tags not in includes_pos
        includes_pos = ["PROPN", "NOUN", "VERB", "ADJ", "ADV"]
        pos_results = tagger.filter_pos(text, includes_pos)
        filtered_text = [pos_result['word'] for pos_result in pos_results]
        text = " ".join(filtered_text).lower()

        text = clean_newlines(text, max_newlines=1)
        text = clean_punctuations(text)
        text = clean_spaces(text)
        text = clean_string(text)

        # Remove stopwords
        words = get_words(text)
        filtered_words = [
            word for word in words if word.lower() not in stop_words]
        text = ' '.join(filtered_words)

        preprocessed_texts[idx] = text

    return preprocessed_texts


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/data/complete_jet_resume.md"
    markdown_tokens = parse_markdown(
        docs_file, merge_headers=False, merge_contents=True, ignore_links=False)
    texts: List[str] = [d["content"] for d in markdown_tokens]

    logger.debug(f"Loaded {len(texts)} documents")

    texts = preprocess_texts(texts)
    save_file(texts, f"{output_dir}/preprocessed_texts.json")

    # Existing function calls
    all_ngrams = count_ngrams([text.lower()
                               for text in texts], min_words=1)
    save_file(all_ngrams, f"{output_dir}/all_ngrams_count.json")

    filtered_ngrams = count_ngrams(
        [text.lower() for text in texts], min_count=2)
    save_file(filtered_ngrams,
              f"{output_dir}/filtered_ngrams_min_count_2.json")

    result = get_most_common_ngrams([text.lower() for text in texts])
    save_file(result, f"{output_dir}/most_common_ngrams.json")

    result = get_common_texts(
        [text.lower() for text in texts], includes_pos=["PROPN", "NOUN", "VERB", "ADJ", "ADV"],)
    save_file(result, f"{output_dir}/common_texts.json")

    result = group_sentences_by_ngram(
        [text.lower() for text in texts], is_start_ngrams=False)
    save_file(result, f"{output_dir}/grouped_sentences.json")

    lowered_ngrams_count = count_ngrams(
        [text.lower() for text in texts], min_words=1, max_words=3)
    save_file(lowered_ngrams_count, f"{output_dir}/lowered_ngrams_count.json")

    range_results = list(get_ngrams_by_range(
        texts, min_words=1, max_words=2, count=(2,), show_count=True))
    save_file(range_results, f"{output_dir}/ngrams_by_range.json")

    count_results = list(get_ngrams_by_range(
        texts, min_words=2, count=2, show_count=True))
    save_file(count_results, f"{output_dir}/ngrams_by_count.json")

    results = filter_texts_by_multi_ngram_count(
        texts, min_words=1, count=(2,), count_all_ngrams=True)
    save_file(results, f"{output_dir}/filtered_texts.json")

    # Calculate n-gram diversity
    ngram_freq = n_gram_frequency(
        " ".join([text.lower() for text in texts]), n=2)
    diversity = calculate_n_gram_diversity(ngram_freq)
    save_file({"diversity": diversity},
              f"{output_dir}/ngram_diversity.json")

    # Separate n-gram lines
    separated_lines = separate_ngram_lines(
        [text.lower() for text in texts], punctuations_split=[',', '/', ':'])
    save_file(separated_lines, f"{output_dir}/separated_ngram_lines.json")

    # Get n-grams
    ngrams_list = get_ngrams([text.lower()
                             for text in texts], min_words=1, max_words=2)
    save_file(ngrams_list, f"{output_dir}/ngrams_list.json")

    # Get n-gram weight for the first sentence
    sentence_ngrams = extract_ngrams(
        texts[0].lower(), min_words=1, max_words=2)
    save_file(sentence_ngrams, f"{output_dir}/extract_ngrams.json")

    previous_ngrams = set()  # Empty for example; adjust based on context
    weight = get_ngram_weight(
        all_ngrams, sentence_ngrams, previous_ngrams)
    save_file({"weight": weight}, f"{output_dir}/ngram_weight.json")

    ngram_results = count_ngrams_with_texts(
        texts=[text.lower() for text in texts],
        min_words=1,
        min_count=2,
        max_words=2
    )
    output_path = f"{output_dir}/ngrams_with_texts.json"
    save_file(ngram_results, output_path)

    # Sort sentences
    sorted_sentences = sort_sentences(
        [text.lower() for text in texts], n=2)
    save_file(sorted_sentences, f"{output_dir}/sorted_sentences.json")

    # Filter and sort sentences by n-grams
    filtered_sorted_sentences = filter_and_sort_sentences_by_ngrams(
        [text.lower() for text in texts], min_words=2, top_n=2, is_start_ngrams=False
    )
    save_file(filtered_sorted_sentences,
              f"{output_dir}/filtered_sorted_sentences.json")

    # Get total unique n-grams
    total_unique = get_total_unique_ngrams(all_ngrams)
    save_file({"total_unique_ngrams": total_unique},
              f"{output_dir}/total_unique_ngrams.json")

    # Get total counts of n-grams
    total_counts = get_total_counts_of_ngrams(all_ngrams)
    save_file({"total_ngram_counts": total_counts},
              f"{output_dir}/total_ngram_counts.json")

    # Get specific n-gram count (using first n-gram as example)
    specific_ngram = next(iter(all_ngrams))
    specific_count = get_specific_ngram_count(
        all_ngrams, specific_ngram)
    save_file(
        {"ngram": specific_ngram, "count": specific_count},
        f"{output_dir}/specific_ngram_count.json"
    )

    # Use nwise to generate sliding window n-grams
    words = texts[0].lower().split()
    nwise_ngrams = [" ".join(ngram) for ngram in nwise(words, n=2)]
    save_file(nwise_ngrams, f"{output_dir}/nwise_ngrams.json")


from itertools import tee, islice
from typing import List
from nltk import word_tokenize, ngrams
import string
from collections import Counter
from typing import Optional, List, Tuple, Union, Dict
from collections import defaultdict, Counter
from collections import defaultdict
import itertools
from words import count_words
from words import get_words
from functools import lru_cache
from tqdm import tqdm
from jet.wordnet.similarity import filter_different_texts


def get_ngrams(text, n=1):
    # Tokenize and filter out punctuation in one step
    words = get_words(text, n)
    return words


def count_ngrams(texts: Union[str, List[str]], n: Union[int, Tuple[int, int]]):
    if isinstance(texts, list):
        texts = "\n".join(texts)

    ngram_counter = Counter()
    n_min = n
    n_max = None

    # Check if n is a tuple
    if isinstance(n, tuple):
        if len(n) == 2:
            n_min, n_max = n
        else:
            n_min = n[0]

    if n_max:
        if n_max > n_min:
            for n_val in range(n_min, n_max + 1):
                ngram_counter.update(Counter(get_ngrams(texts, n_val)))
    else:
        ngram_counter.update(Counter(get_ngrams(texts, n_min)))

    return ngram_counter


def filter_sentences_by_pos_tags(sentences, pos_tags):
    print("Filtering sentences by POS tags...")


def group_sentences_by_ngram(
        sentences: list,
        n: int = 2,
        top_n: int = 2,
        is_start_ngrams: bool = True) -> dict:
    sentence_ngrams = defaultdict(list)
    for sentence in tqdm(sentences, desc="Grouping sentences"):
        ngrams_list = get_ngrams(sentence, n)
        if is_start_ngrams and ngrams_list:
            sentence_ngrams[" ".join(ngrams_list[0])].append(sentence)
        elif not is_start_ngrams:
            unique_ngrams = set(ngrams_list)
            for ngram in unique_ngrams:
                sentence_ngrams[" ".join(ngram)].append(sentence)

    optimized_groups = {}
    for ngram, group_sentences in tqdm(sentence_ngrams.items(), desc="Optimizing groups"):
        sorted_group_sentences = sorted(
            group_sentences, key=lambda s: (count_words(s), sentences.index(s)))
        optimized_groups[ngram] = sorted_group_sentences[:top_n]

    return optimized_groups


def n_gram_frequency(sentence, n=2):
    """ Calculate the frequency of n-grams in a sentence """
    n_grams = [sentence[i:i+n] for i in range(len(sentence) - n + 1)]
    return Counter(n_grams)


def calculate_n_gram_diversity(freq):
    """ Calculate diversity based on the count of unique n-grams """
    return len(freq)


def get_ngram_weight(all_ngrams, sentence_ngrams, previous_ngrams):
    # Calculate weight of the sentence based on n-gram frequency
    # Introduce penalty for shared n-grams with the previous sentence
    penalty = sum(ngram in previous_ngrams for ngram in sentence_ngrams)
    return sum(1 / all_ngrams[ngram] for ngram in sentence_ngrams if ngram in all_ngrams) + penalty


def sort_sentences(sentences, n):
    all_ngrams = Counter()
    sentence_ngrams_dict = {}

    # Precompute n-grams for each sentence
    for sentence in tqdm(sentences, desc="Precomputing n-grams"):
        ngram_list = get_ngrams(sentence, n)
        all_ngrams.update(ngram_list)
        sentence_ngrams_dict[sentence] = ngram_list

    sorted_sentences = []

    # Adding tqdm progress bar
    for _ in tqdm(range(len(sentences)), desc="Sorting sentences"):
        if sorted_sentences:
            previous_ngrams = set(get_ngrams(sorted_sentences[-1], n))
        else:
            previous_ngrams = set()

        # Sort remaining sentences based on n-gram weights and penalties
        sentences.sort(key=lambda sentence: get_ngram_weight(
            all_ngrams, sentence_ngrams_dict[sentence], previous_ngrams),
            reverse=False
        )

        # Add the best sentence to the sorted list and remove it from the original list
        sorted_sentences.append(sentences.pop(0))

    return sorted_sentences


def filter_and_sort_sentences_by_ngrams(sentences: List[str], n: int = 2, top_n: int = 2, is_start_ngrams=True) -> List[str]:
    sentence_ngrams = defaultdict(list)
    all_ngrams = Counter()

    # Combine grouping and ngram counting in a single loop
    for sentence in tqdm(sentences, desc="Grouping sentences"):
        ngrams_list = get_ngrams(sentence, n)
        all_ngrams.update(ngrams_list)

        if is_start_ngrams and ngrams_list:
            sentence_ngrams[" ".join(ngrams_list[0])].append(sentence)
        elif not is_start_ngrams:
            for ngram in set(ngrams_list):
                sentence_ngrams[" ".join(ngram)].append(sentence)

    # Optimizing groups without a secondary sorting loop
    optimized_groups = {ngram: group_sentences[:top_n]
                        for ngram, group_sentences in sentence_ngrams.items()}

    # Flatten the dictionary of grouped sentences
    flattened_sentences = set(
        itertools.chain.from_iterable(optimized_groups.values()))

    # Sort sentences by unique ngram weights
    sorted_sentences = sort_sentences(list(flattened_sentences), n)

    return sorted_sentences


def filter_and_sort_sentences_by_similarity(sentences: List[str], n=2, threshold=0.8) -> List[str]:
    filtered_sentences = filter_different_texts(sentences, threshold)
    sorted_sentences = sort_sentences(filtered_sentences, n)
    return sorted_sentences


def recursive_filter(texts: List[str], max_n: int, depth: int = 0, max_depth: int = 10) -> Tuple[List[str], List[str]]:
    if depth > max_depth:
        # Prevent too deep recursion
        return texts, []

    ngram_counter = Counter()
    text_ngram_counts = {}

    for text in texts:
        text_ngrams = count_ngrams(text, max_n)
        text_ngram_counts[text] = text_ngrams
        ngram_counter.update(text_ngrams)

    passed_texts = []
    failed_texts = []

    for text, ngrams in text_ngram_counts.items():
        if all(ngram_counter[ngram] <= max_n for ngram in ngrams):
            passed_texts.append(text)
        else:
            failed_texts.append(text)

    if failed_texts:
        # Recursive call with the failed texts
        additional_passed, _ = recursive_filter(
            failed_texts, max_n, depth + 1, max_depth)
        passed_texts.extend(additional_passed)

    return passed_texts, failed_texts


def get_most_common_ngrams(texts: str | List[str], min_count: int = 2, n: Union[int, Tuple[int, int]] = (2,)) -> dict:
    if isinstance(texts, list):
        text = " ".join(texts)
    else:
        text = texts
    n_min = n[0] if isinstance(n, tuple) else n
    most_common_results = []

    while True:
        if isinstance(n, tuple) and len(n) == 2:
            n_max = n[1]
            ngram_counter = count_ngrams(text, n=(n_min, n_max))
        else:
            ngram_counter = count_ngrams(text, n=(n_min,))
        filtered_results = [(ngram, count) for ngram,
                            count in ngram_counter.items() if count >= min_count]
        if filtered_results:
            most_common_results.extend(filtered_results)
            n_min += 1
        else:
            break

    # Filter out n-grams that are substrings of other n-grams
    filtered_results = []
    for ngram, count in most_common_results:
        is_substring = False
        for other_ngram, other_count in most_common_results:
            if ngram != other_ngram and count == other_count and other_ngram.startswith(ngram):
                is_substring = True
                break
        if not is_substring:
            filtered_results.append((ngram, count))

    # Sort by count descending, then by ngram ascending
    filtered_results.sort(key=lambda x: (-x[1], x[0]))
    return dict(filtered_results)


def get_total_unique_ngrams(ngram_counter):
    return len(ngram_counter)


def get_total_counts_of_ngrams(ngram_counter):
    return sum(ngram_counter.values())


def get_specific_ngram_count(ngram_counter, specific_ngram):
    return ngram_counter[specific_ngram]


def get_ngrams_by_range(
    texts: Union[str, List[str]],
    n: Union[int, Tuple[int, int]],
    count: Optional[Union[int, Tuple[int, int]]] = None,
    show_count: bool = False
) -> Union[List[str], List[Dict[str, int]]]:
    is_texts_list = isinstance(texts, list)
    if is_texts_list:
        texts = " ".join(texts)

    ngram_counter = count_ngrams(texts, n)

    if is_texts_list:
        pbar = tqdm(ngram_counter.items(), desc="Processing ngrams")
    else:
        pbar = ngram_counter.items()

    results = []

    def add_result(ngram, count):
        if show_count:
            results.append({"ngram": ngram, "count": count})
        else:
            results.append(ngram)

    for ngram, n_count in pbar:
        if isinstance(count, tuple):
            count_min, count_max = (count[0], float(
                'inf')) if len(count) == 1 else count
            if count_min <= n_count <= count_max:
                add_result(ngram, n_count)
        elif isinstance(count, int) and count <= n_count:
            add_result(ngram, n_count)
        elif count is None:
            add_result(ngram, n_count)

    return results


def filter_texts_by_multi_ngram_count(
        texts: List[str],
        n: int,
        count: Union[int, Tuple[int, int]],
        count_all_ngrams: bool = True,
) -> List[str]:
    ngrams_list = get_ngrams_by_range(texts, n, count, show_count=True)
    ngrams_dict = {ngram_dict['ngram']: ngram_dict['count']
                   for ngram_dict in ngrams_list}

    if isinstance(count, tuple):
        count_min, count_max = (count[0], float(
            'inf')) if len(count) == 1 else count
    else:
        count_min = count
        count_max = float('inf')

    filtered_texts = []

    if count_all_ngrams:
        for text in tqdm(texts, desc="Filtering texts"):
            text_words = get_words(text, n=n)
            # Check if all text_words exist in ngrams_dict
            if all(word in ngrams_dict for word in text_words):
                filtered_texts.append(text)
    else:
        for text in tqdm(texts, desc="Filtering texts"):
            # match_ngrams_dict are all the ngrams in the text
            match_ngrams_dict = {
                ngram: ngrams_dict[ngram] for ngram in ngrams_dict if ngram in text}

            # if one matched ngram has a count that violates the count range, skip the text
            n_counts = match_ngrams_dict.values()
            has_violating_count = False
            for n_count in n_counts:
                if not count_min <= n_count <= count_max:
                    has_violating_count = True
                    break
            # if no ngrams in the text match the ngrams_dict, skip the text
            if has_violating_count or not match_ngrams_dict:
                continue
            # if all matched ngrams have a count that is within the count range, add the text
            filtered_texts.append(text)

    return filtered_texts


def nwise(iterable, n=1):
    "Returns a sliding window (of width n) over data from the iterable"
    iters = tee(iterable, n)
    for i, it in enumerate(iters):
        next(islice(it, i, i), None)
    return zip(*iters)


if __name__ == "__main__":
    texts = [
        "Ilarawan ang istruktura na mahalaga",
        "Magbigay ng tatlong tip",
        "Paano natin mababawasan?",
        "Kailangan mong gumawa ng isang mahirap na desisyon.",
        "Kilalanin ang kakaiba.",
        "Ipaliwanag kung bakit",
        "Sumulat ng isang maikling kuwento",
        "Ilarawan ang istruktura sa buhok",
        "Dahil ang istruktura na mahalaga",
    ]

    # stopwords = StopWords()
    # Remove stopwords
    # texts = [stopwords.remove_stop_words(text, 'tagalog') for text in texts]

    text = " ".join(texts)

    # All n-grams count
    print("\nAll n-grams:")
    ngram_counter = count_ngrams(text.lower(), n=1)
    for ngram, count in ngram_counter.items():
        print(f"{ngram}: {count}")

    print("\nMost Common n-grams:")
    print(get_most_common_ngrams(texts))

    # Specific n-grams count
    ngram_counter = count_ngrams(text, (1, 4))

    print(f"\nTotal unique n-grams: {get_total_unique_ngrams(ngram_counter)}")
    print(
        f"\nTotal counts of all n-grams: {get_total_counts_of_ngrams(ngram_counter)}")

    specific_ngram = ('This', 'is')  # Example specific n-gram
    print(
        f"\nCount of {specific_ngram}: {get_specific_ngram_count(ngram_counter, specific_ngram)}")

    print("\nN-grams of Specific Range:")
    for ngram_dict in get_ngrams_by_range(texts, n=(1, 2), count=(2, ), show_count=True):
        print(f"{ngram_dict['ngram']}: {ngram_dict['count']}")

    print("\nN-grams of Specific Count:")
    for ngram in get_ngrams_by_range(texts, n=2, count=2, show_count=True):
        print(f"{ngram}")

    results = filter_texts_by_multi_ngram_count(
        texts, n=1, count=(2, ), count_all_ngrams=True)
    print(
        f"\nFilter texts by multi-ngram count: {len(results)}\nOriginal: {len(texts)}")
    print(results)

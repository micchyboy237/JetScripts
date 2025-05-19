from sklearn.model_selection import train_test_split
from jet.logger import time_it
from .words import get_unique_words, get_words
from typing import List, Optional
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import itertools


class ProcessedData():
    source: str
    target: Optional[str]
    category_values: List[str]
    score: Optional[float]


def get_ngrams(text, n=1):
    # Tokenize and filter out punctuation in one step
    words = get_words(text, n)
    return words


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


def n_gram_frequency(sentence, n=2):
    """ Calculate the frequency of n-grams in a sentence """
    n_grams = [sentence[i:i+n] for i in range(len(sentence) - n + 1)]
    return Counter(n_grams)


def calculate_n_gram_diversity(freq):
    """ Calculate diversity based on the count of unique n-grams """
    return len(freq)


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


class ProcessedDataString():
    source: str
    category_values: List[str]


class StratifiedData():
    def __init__(self, source: str, target: str, score: float):
        self.source = source
        self.target = target
        self.score = score


class StratifiedSampler():
    def __init__(self, data: List[ProcessedData or str], num_samples=0.8):
        # Assuming get_unique_words is defined elsewhere
        unique_words = get_unique_words(data)
        self.data = unique_words

        # Calculate the number of samples
        if isinstance(num_samples, float) and 0.0 < num_samples < 1.0:
            final_num_samples = int(num_samples * len(data))
        elif isinstance(num_samples, int) and num_samples > 0:
            final_num_samples = min(num_samples, len(data))
        else:
            raise ValueError(
                "num_samples must be a float in the range (0.0, 1.0) or a positive integer")

        self.num_samples = final_num_samples

    def filter_strings(self, n=2, top_n=2) -> List[str]:
        filtered_data = filter_and_sort_sentences_by_ngrams(
            self.data, n, top_n, is_start_ngrams=True)

        return filtered_data[:self.num_samples]

    @time_it
    def get_samples(self) -> List[StratifiedData]:
        # Create a dictionary to map (source, target) tuples to their scores
        score_map = {(item.source, item.target): item.score for item in self.data}

        # Unpack the data into features, targets, and labels for stratification
        features_targets = [(item.source, item.target) for item in self.data]
        labels = [item.category_values for item in self.data]

        # Use labels for stratification
        features_targets_sample, _, labels_sample, _ = train_test_split(
            features_targets, labels, train_size=self.num_samples, stratify=labels
        )

        # Construct StratifiedData objects from the sampled data, including the score
        stratified_samples = [
            StratifiedData(source=ft[0], target=ft[1], score=score_map[ft])
            for ft, lbl in zip(features_targets_sample, labels_sample)
        ]

        # Convert all items to dict
        stratified_samples = [item.__dict__ for item in stratified_samples]

        return stratified_samples

    @time_it
    def get_unique_strings(self) -> List[str]:
        data_with_labels = self.load_data_with_labels()

        # Unpack the data into features, targets, and labels for stratification
        features_targets = [item.source for item in data_with_labels]
        # Now includes n-gram categories
        labels = [item.category_values for item in data_with_labels]

        features_targets_sample, _, labels_sample, _ = train_test_split(
            features_targets, labels, train_size=self.num_samples, stratify=labels
        )

        return features_targets_sample

    @time_it
    def load_data_with_labels(
        self,
        max_q=2,
    ) -> List[ProcessedDataString]:
        data = self.data

        def calculate_ttr(sentence):
            words = sentence.split()
            unique_words = set(words)
            return len(unique_words)

        def calculate_ttr_class(ttr, ttr_quantiles):
            for i, q in enumerate(ttr_quantiles):
                if ttr <= q:
                    return f'ttr_q{i+1}'
            return f'ttr_q{len(ttr_quantiles)+1}'

        def categorize_sentence_length(sentence, length_quantiles):
            word_count = len(sentence.split())
            for i, q in enumerate(length_quantiles):
                if word_count <= q:
                    return f'q{i+1}'
            return f'q{len(length_quantiles)+1}'

        def categorize_n_gram_diversity(diversity, diversity_quantiles):
            """ Categorize based on n-gram diversity quantiles """
            for i, q in enumerate(diversity_quantiles):
                if diversity <= q:
                    return f'ngram_q{i+1}'
            return f'ngram_q{len(diversity_quantiles)+1}'

        def get_starting_n_gram(sentence, n=5):
            """ Extract the starting n-gram of a sentence (consider increasing 'n' for more granularity) """
            words = get_words(sentence)
            return ' '.join(words[:n]) if len(words) >= n else sentence

        def categorize_based_on_quantiles(value, quantiles):
            """ Categorize a value based on quantiles """
            for i, q in enumerate(quantiles):
                if value <= q:
                    return f'q{i+1}'
            return f'q{len(quantiles)+1}'

        def determine_quantiles(values, num_quantiles):
            """ Determine dynamic quantile values based on the data distribution """
            quantile_values = np.linspace(0, 1, num_quantiles + 2)[1:-1]
            return np.quantile(values, quantile_values)

        # Compute quantiles for dynamic categorization
        sentence_counts = [len(item.split()) for item in data]
        ttrs = [calculate_ttr(item) for item in data]

        # Determine number of quantiles based on data diversity and max_q
        num_length_quantiles = min(max_q, min(
            5, len(set(sentence_counts)) // 20))
        num_ttr_quantiles = min(max_q, min(5, len(set(ttrs)) // 20))

        length_quantiles = determine_quantiles(
            sentence_counts, num_length_quantiles)
        ttr_quantiles = determine_quantiles(ttrs, num_ttr_quantiles)

        # Compute n-gram frequencies and determine their diversity
        ngram_diversities = [calculate_n_gram_diversity(
            n_gram_frequency(item)) for item in data]

        # Determine number of quantiles for n-gram diversity
        num_ngram_quantiles = min(max_q, min(
            5, len(set(ngram_diversities)) // 20))
        ngram_quantiles = determine_quantiles(
            ngram_diversities, num_ngram_quantiles)

       # Compute starting n-gram frequencies
        starting_ngrams = [get_starting_n_gram(item) for item in data]
        starting_ngram_freq = Counter(starting_ngrams)
        starting_ngram_counts = list(starting_ngram_freq.values())

        # Determine dynamic quantiles for starting n-gram frequencies
        num_starting_ngram_quantiles = min(max_q, min(
            5, len(set(starting_ngram_counts)) // 20))
        starting_ngram_quantiles = determine_quantiles(
            starting_ngram_counts, num_starting_ngram_quantiles)

        # Categorize each starting n-gram based on quantiles
        starting_ngram_categories = {}
        for ngram in starting_ngram_freq:
            ngram_count = starting_ngram_freq[ngram]
            starting_ngram_category = categorize_based_on_quantiles(
                ngram_count, starting_ngram_quantiles)
            starting_ngram_categories[ngram] = starting_ngram_category

        processed_data = []
        ttr_class_distribution = Counter()
        sentence_length_distribution = Counter()
        n_gram_diversity_distribution = Counter()
        starting_ngram_distribution = Counter()

        for item in data:
            source_sentence = item

            ttr = calculate_ttr(source_sentence)
            ttr_class = calculate_ttr_class(ttr, ttr_quantiles)
            sentence_length = categorize_sentence_length(
                source_sentence, length_quantiles)

            n_gram_diversity = calculate_n_gram_diversity(
                n_gram_frequency(source_sentence))
            n_gram_diversity_class = categorize_n_gram_diversity(
                n_gram_diversity, ngram_quantiles)

            # Get starting n-gram category
            starting_ngram = get_starting_n_gram(source_sentence)
            starting_ngram_category = starting_ngram_categories[starting_ngram]

            # Update distributions
            ttr_class_distribution[ttr_class] += 1
            sentence_length_distribution[sentence_length] += 1
            n_gram_diversity_distribution[n_gram_diversity_class] += 1
            starting_ngram_distribution[starting_ngram_category] += 1

            # Create a new processed data item
            processed_item = ProcessedDataString()
            processed_item.source = source_sentence
            processed_item.category_values = [
                ttr_class, sentence_length, n_gram_diversity_class, starting_ngram_category]

            processed_data.append(processed_item)

        # Print out the distributions
        print("TTR Class Distribution:", dict(ttr_class_distribution))
        print("Sentence Length Distribution:",
              dict(sentence_length_distribution))
        print("N-Gram Diversity Distribution:",
              dict(n_gram_diversity_distribution))
        print("Starting N-Gram Distribution:",
              dict(starting_ngram_distribution))

        return processed_data

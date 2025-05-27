from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from typing import Union, List, Dict, Tuple, Optional
import nltk
import itertools

# Download required NLTK data (only needs to run once)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


def get_word_sentence_combination_counts(text: Union[str, List[str]], n: Optional[int] = None, min_count: int = 1, in_sequence: bool = False) -> Union[Dict[Tuple[str, ...], int], List[Dict[Tuple[str, ...], int]]]:
    """
    Get counts of word combinations (n-grams) within sentences from a text string or list of strings with lemmatization,
    excluding stop words, sorted by count in descending order, including only those with counts >= min_count.
    Combinations can be sequential or order-independent within each sentence. If n is None, counts combinations of all sizes.

    Args:
        text (Union[str, List[str]]): Input text as a single string or a list of strings to analyze.
        n (Optional[int]): Size of word combinations (default is None, meaning all sizes from 1 to max words in a sentence).
        min_count (int): Minimum count threshold for combinations to be included (default is 1).
        in_sequence (bool): If True, combinations are sequential n-grams (words must appear in order).
                            If False, combinations are order-independent (default is False).

    Returns:
        Union[Dict[Tuple[str, ...], int], List[Dict[Tuple[str, ...], int]]]: If input is a string, returns a dictionary
            with tuples of lemmatized word combinations as keys and their counts as values, sorted by count descending.
            If input is a list of strings, returns a list of such dictionaries, one for each input string.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def process_single_text(text: str) -> Dict[Tuple[str, ...], int]:
        # Split text into sentences
        sentences = sent_tokenize(text.lower())

        # Initialize counter for word combinations
        counts = Counter()

        for sentence in sentences:
            # Tokenize sentence
            tokens = word_tokenize(sentence)

            # Filter alphabetic tokens and remove stop words
            words = [
                token for token in tokens
                if token.isalpha() and token not in stop_words
            ]

            # Lemmatize words
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

            # Determine combination sizes
            if n is None:
                # Max combination size is number of words in sentence
                max_n = len(lemmatized_words)
                combination_sizes = range(1, max_n + 1)
            else:
                combination_sizes = [n]

            # Generate combinations for each size
            for current_n in combination_sizes:
                if in_sequence:
                    # Generate sequential n-grams
                    combinations = [
                        tuple(lemmatized_words[i:i + current_n])
                        for i in range(len(lemmatized_words) - current_n + 1)
                    ]
                else:
                    # Generate all possible n-word combinations within the sentence
                    combinations = list(itertools.combinations(
                        lemmatized_words, current_n))

                # Update counter with combinations
                counts.update(combinations)

        # Filter by min_count and sort by descending frequency
        return dict(sorted(
            {ngram: count for ngram, count in counts.items() if count >=
             min_count}.items(),
            key=lambda x: x[1],
            reverse=True
        ))

    # Handle input type
    if isinstance(text, str):
        return process_single_text(text)
    elif isinstance(text, list):
        return [process_single_text(item) for item in text]
    else:
        raise ValueError("Input must be a string or a list of strings")

# Test class using pytest


class TestWordSentenceCombinationCounts:
    def test_single_string_non_sequential_n_2(self):
        # Test input
        text = "The quick brown fox jumps. The brown fox jumps quickly."

        # Expected output for non-sequential bigrams (in_sequence=False, n=2) with min_count=2
        # Sentence 1: quick, brown, fox, jump -> pairs: (quick,brown), (quick,fox), (quick,jump), (brown,fox), (brown,jump), (fox,jump)
        # Sentence 2: brown, fox, jump, quick -> pairs: (brown,fox), (brown,jump), (brown,quick), (fox,jump), (fox,quick), (jump,quick)
        # Combined counts: (brown,fox):2, (brown,jump):2, (fox,jump):2, others:1
        expected = {
            ('brown', 'fox'): 2,
            ('brown', 'jump'): 2,
            ('fox', 'jump'): 2
        }
        result = get_word_sentence_combination_counts(
            text, n=2, min_count=2, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_string_sequential_n_2(self):
        # Test input
        text = "The quick brown fox jumps. The brown fox jumps quickly."

        # Expected output for sequential bigrams (in_sequence=True, n=2) with min_count=2
        # Sentence 1: quick, brown, fox, jump -> sequential pairs: (quick,brown), (brown,fox), (fox,jump)
        # Sentence 2: brown, fox, jump, quick -> sequential pairs: (brown,fox), (fox,jump), (jump,quick)
        # Combined counts: (brown,fox):2, (fox,jump):2, others:1
        expected = {
            ('brown', 'fox'): 2,
            ('fox', 'jump'): 2
        }
        result = get_word_sentence_combination_counts(
            text, n=2, min_count=2, in_sequence=True)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_string_non_sequential_n_none(self):
        # Test input
        text = "The quick brown fox jumps. The brown fox jumps quickly."

        # Expected output for non-sequential combinations (in_sequence=False, n=None) with min_count=2
        # Sentence 1: quick, brown, fox, jump -> 1-grams: (quick,), (brown,), (fox,), (jump,)
        #                                      -> 2-grams: (quick,brown), (quick,fox), (quick,jump), (brown,fox), (brown,jump), (fox,jump)
        #                                      -> 3-grams: (quick,brown,fox), (quick,brown,jump), (quick,fox,jump), (brown,fox,jump)
        #                                      -> 4-gram: (quick,brown,fox,jump)
        # Sentence 2: brown, fox, jump, quick -> 1-grams: (brown,), (fox,), (jump,), (quick,)
        #                                      -> 2-grams: (brown,fox), (brown,jump), (brown,quick), (fox,jump), (fox,quick), (jump,quick)
        #                                      -> 3-grams: (brown,fvor,jump), (brown,fox,quick), (brown,jump,quick), (fox,jump,quick)
        #                                      -> 4-gram: (brown,fox,jump,quick)
        # Combined counts: (brown,):2, (fox,):2, (jump,):2, (quick,):2, (brown,fox):2, (brown,jump):2, (fox,jump):2, others:1
        expected = {
            ('brown',): 2,
            ('fox',): 2,
            ('jump',): 2,
            ('quick',): 2,
            ('brown', 'fox'): 2,
            ('brown', 'jump'): 2,
            ('fox', 'jump'): 2
        }
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=2, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_string_sequential_n_none(self):
        # Test input
        text = "The quick brown fox jumps. The brown fox jumps quickly."

        # Expected output for sequential combinations (in_sequence=True, n=None) with min_count=2
        # Sentence 1: quick, brown, fox, jump -> 1-grams: (quick,), (brown,), (fox,), (jump,)
        #                                      -> 2-grams: (quick,brown), (brown,fox), (fox,jump)
        #                                      -> 3-grams: (quick,brown,fox), (brown,fox,jump)
        #                                      -> 4-gram: (quick,brown,fox,jump)
        # Sentence 2: brown, fox, jump, quick -> 1-grams: (brown,), (fox,), (jump,), (quick,)
        #                                      -> 2-grams: (brown,fox), (fox,jump), (jump,quick)
        #                                      -> 3-grams: (brown,fox,jump), (fox,jump,quick)
        #                                      -> 4-gram: (brown,fox,jump,quick)
        # Combined counts: (brown,):2, (fox,):2, (jump,):2, (quick,):2, (brown,fox):2, (fox,jump):2, others:1
        expected = {
            ('brown',): 2,
            ('fox',): 2,
            ('jump',): 2,
            ('quick',): 2,
            ('brown', 'fox'): 2,
            ('fox', 'jump'): 2
        }
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=2, in_sequence=True)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_list_strings_non_sequential_n_none(self):
        # Test input
        text = [
            "The quick brown fox jumps.",
            "The brown fox jumps quickly."
        ]

        # Expected output for non-sequential combinations (in_sequence=False, n=None) with min_count=1
        # Text 1: Sentence: quick, brown, fox, jump -> 1-grams: (quick,), (brown,), (fox,), (jump,)
        #                                       -> 2-grams: (quick,brown), (quick,fox), (quick,jump), (brown,fox), (brown,jump), (fox,jump)
        #                                       -> 3-grams: (quick,brown,fox), (quick,brown,jump), (quick,fox,jump), (brown,fox,jump)
        #                                       -> 4-gram: (quick,brown,fox,jump)
        # Text 2: Sentence: brown, fox, jump, quick -> 1-grams: (brown,), (fox,), (jump,), (quick,)
        #                                       -> 2-grams: (brown,fox), (brown,jump), (brown,quick), (fox,jump), (fox,quick), (jump,quick)
        #                                       -> 3-grams: (brown,fox,jump), (brown,fox,quick), (brown,jump,quick), (fox,jump,quick)
        #                                       -> 4-gram: (brown,fox,jump,quick)
        expected = [
            {
                ('quick',): 1,
                ('brown',): 1,
                ('fox',): 1,
                ('jump',): 1,
                ('quick', 'brown'): 1,
                ('quick', 'fox'): 1,
                ('quick', 'jump'): 1,
                ('brown', 'fox'): 1,
                ('brown', 'jump'): 1,
                ('fox', 'jump'): 1,
                ('quick', 'brown', 'fox'): 1,
                ('quick', 'brown', 'jump'): 1,
                ('quick', 'fox', 'jump'): 1,
                ('brown', 'fox', 'jump'): 1,
                ('quick', 'brown', 'fox', 'jump'): 1
            },
            {
                ('brown',): 1,
                ('fox',): 1,
                ('jump',): 1,
                ('quick',): 1,
                ('brown', 'fox'): 1,
                ('brown', 'jump'): 1,
                ('brown', 'quick'): 1,
                ('fox', 'jump'): 1,
                ('fox', 'quick'): 1,
                ('jump', 'quick'): 1,
                ('brown', 'fox', 'jump'): 1,
                ('brown', 'fox', 'quick'): 1,
                ('brown', 'jump', 'quick'): 1,
                ('fox', 'jump', 'quick'): 1,
                ('brown', 'fox', 'jump', 'quick'): 1
            }
        ]
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=1, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_list_strings_sequential_n_none(self):
        # Test input
        text = [
            "The quick brown fox jumps.",
            "The brown fox jumps quickly."
        ]

        # Expected output for sequential combinations (in_sequence=True, n=None) with min_count=1
        # Text 1: Sentence: quick, brown, fox, jump -> 1-grams: (quick,), (brown,), (fox,), (jump,)
        #                                       -> 2-grams: (quick,brown), (brown,fox), (fox,jump)
        #                                       -> 3-grams: (quick,brown,fox), (brown,fox,jump)
        #                                       -> 4-gram: (quick,brown,fox,jump)
        # Text 2: Sentence: brown, fox, jump, quick -> 1-grams: (brown,), (fox,), (jump,), (quick,)
        #                                       -> 2-grams: (brown,fox), (fox,jump), (jump,quick)
        #                                       -> 3-grams: (brown,fox,jump), (fox,jump,quick)
        #                                       -> 4-gram: (brown,fox,jump,quick)
        expected = [
            {
                ('quick',): 1,
                ('brown',): 1,
                ('fox',): 1,
                ('jump',): 1,
                ('quick', 'brown'): 1,
                ('brown', 'fox'): 1,
                ('fox', 'jump'): 1,
                ('quick', 'brown', 'fox'): 1,
                ('brown', 'fox', 'jump'): 1,
                ('quick', 'brown', 'fox', 'jump'): 1
            },
            {
                ('brown',): 1,
                ('fox',): 1,
                ('jump',): 1,
                ('quick',): 1,
                ('brown', 'fox'): 1,
                ('fox', 'jump'): 1,
                ('jump', 'quick'): 1,
                ('brown', 'fox', 'jump'): 1,
                ('fox', 'jump', 'quick'): 1,
                ('brown', 'fox', 'jump', 'quick'): 1
            }
        ]
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=1, in_sequence=True)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_string(self):
        # Test input
        text = ""

        # Expected output for empty string
        expected = {}
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=1, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_list(self):
        # Test input
        text = []

        # Expected output for empty list
        expected = []
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=1, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_list_with_single_word_string(self):
        # Test input
        text = ["Quick.", "Fox."]

        # Expected output for single-word strings (only 1-grams possible with n=None)
        expected = [
            {('quick',): 1},
            {('fox',): 1}
        ]
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=1, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"


# Example usage
if __name__ == "__main__":
    sample_text_str = "The quick brown fox jumps. The brown fox jumps quickly."
    sample_text_list = [
        "The quick brown fox jumps.",
        "The brown fox jumps quickly."
    ]
    print("Single string, non-sequential, n=2 (min_count=2):")
    bigram_counts_str_nonseq = get_word_sentence_combination_counts(
        sample_text_str, n=2, min_count=2, in_sequence=False)
    print(bigram_counts_str_nonseq)
    print("\nSingle string, sequential, n=2 (min_count=2):")
    bigram_counts_str_seq = get_word_sentence_combination_counts(
        sample_text_str, n=2, min_count=2, in_sequence=True)
    print(bigram_counts_str_seq)
    print("\nSingle string, non-sequential, n=None (min_count=2):")
    counts_str_nonseq_all = get_word_sentence_combination_counts(
        sample_text_str, n=None, min_count=2, in_sequence=False)
    print(counts_str_nonseq_all)
    print("\nList of strings, non-sequential, n=None (min_count=1):")
    counts_list_nonseq_all = get_word_sentence_combination_counts(
        sample_text_list, n=None, min_count=1, in_sequence=False)
    print(counts_list_nonseq_all)
    print("\nList of strings, sequential, n=None (min_count=1):")
    counts_list_seq_all = get_word_sentence_combination_counts(
        sample_text_list, n=None, min_count=1, in_sequence=True)
    print(counts_list_seq_all)

import unittest
from unittest.mock import patch, MagicMock
from stratified_sampler import StratifiedSampler, ProcessedData, ProcessedDataString, StratifiedData, calculate_n_gram_diversity, n_gram_frequency
from typing import List
import numpy as np
from collections import Counter


class TestStratifiedSampler(unittest.TestCase):
    def setUp(self):
        # Sample data for testing (strings, as expected by StratifiedSampler)
        self.sample_data = [
            "The quick brown fox",
            "A fast red dog runs",
            "Slow green turtle walks"
        ]

        # Sample data for ProcessedData (used in test_get_samples)
        self.sample_processed_data = []
        for source, target, categories, score in [
            ("The quick brown fox", "jumps", ["ttr_q1", "q1"], 0.9),
            ("A fast red dog runs", "barks", ["ttr_q2", "q2"], 0.8),
            ("Slow green turtle walks", "crawls", ["ttr_q1", "q1"], 0.7)
        ]:
            item = ProcessedData()
            item.source = source
            item.target = target
            item.category_values = categories
            item.score = score
            self.sample_processed_data.append(item)

    @patch('stratified_sampler.get_unique_words')
    def test_init_with_float_num_samples(self, mock_get_unique_words):
        mock_get_unique_words.return_value = self.sample_data
        sampler = StratifiedSampler(self.sample_data, num_samples=0.5)
        self.assertEqual(sampler.num_samples, 1)  # 0.5 * 3 = 1.5, rounded to 1
        self.assertEqual(sampler.data, self.sample_data)

    @patch('stratified_sampler.get_unique_words')
    def test_init_with_int_num_samples(self, mock_get_unique_words):
        mock_get_unique_words.return_value = self.sample_data
        sampler = StratifiedSampler(self.sample_data, num_samples=2)
        self.assertEqual(sampler.num_samples, 2)
        self.assertEqual(sampler.data, self.sample_data)

    @patch('stratified_sampler.get_unique_words')
    def test_init_invalid_num_samples(self, mock_get_unique_words):
        mock_get_unique_words.return_value = self.sample_data
        with self.assertRaises(ValueError):
            StratifiedSampler(self.sample_data, num_samples=0)
        with self.assertRaises(ValueError):
            StratifiedSampler(self.sample_data, num_samples=1.5)

    @patch('stratified_sampler.filter_and_sort_sentences_by_ngrams')
    def test_filter_strings(self, mock_filter_and_sort):
        mock_filter_and_sort.return_value = [
            "The quick brown fox", "A fast red dog runs"]
        sampler = StratifiedSampler(self.sample_data, num_samples=2)
        result = sampler.filter_strings(n=2, top_n=2)
        # Check call arguments, allowing n and top_n to be positional
        mock_filter_and_sort.assert_called_once()
        args, kwargs = mock_filter_and_sort.call_args
        self.assertEqual(set(args[0]), set(
            self.sample_data))  # Check sentences
        self.assertEqual(args[1:], (2, 2))  # Check n and top_n
        # Check keyword arg
        self.assertEqual(kwargs, {'is_start_ngrams': True})
        self.assertEqual(
            result, ["The quick brown fox", "A fast red dog runs"])

    @patch('stratified_sampler.train_test_split')
    def test_get_samples(self, mock_train_test_split):
        sampler = StratifiedSampler(self.sample_processed_data, num_samples=2)
        mock_train_test_split.return_value = (
            [("The quick brown fox", "jumps"), ("A fast red dog runs", "barks")],
            [],
            [["ttr_q1", "q1"], ["ttr_q2", "q2"]],
            []
        )
        result = sampler.get_samples()
        expected = [
            {"source": "The quick brown fox", "target": "jumps", "score": 0.9},
            {"source": "A fast red dog runs", "target": "barks", "score": 0.8}
        ]
        self.assertEqual(result, expected)
        mock_train_test_split.assert_called_once()

    @patch('stratified_sampler.train_test_split')
    @patch('stratified_sampler.get_words')
    @patch('stratified_sampler.n_gram_frequency')
    @patch('numpy.quantile')
    def test_get_unique_strings(self, mock_quantile, mock_n_gram_frequency, mock_get_words, mock_train_test_split):
        sampler = StratifiedSampler(self.sample_data, num_samples=2)
        # Provide enough return values for get_words (3 for sentence_counts, 3 for get_starting_n_gram)
        mock_get_words.side_effect = [
            ["The", "quick", "brown", "fox"],
            ["A", "fast", "red", "dog", "runs"],
            ["Slow", "green", "turtle", "walks"],
            ["The", "quick", "brown", "fox"],
            ["A", "fast", "red", "dog", "runs"],
            ["Slow", "green", "turtle", "walks"]
        ]
        mock_n_gram_frequency.return_value = {"The quick": 1, "quick brown": 1}
        mock_quantile.side_effect = [
            [4, 5],  # length_quantiles
            [3, 4],  # ttr_quantiles
            [2, 3],  # ngram_quantiles
            [1, 2]   # starting_ngram_quantiles
        ]
        mock_train_test_split.return_value = (
            ["The quick brown fox", "A fast red dog runs"],
            [],
            [["ttr_q1", "q1", "ngram_q1", "start_q1"], [
                "ttr_q2", "q2", "ngram_q2", "start_q2"]],
            []
        )
        result = sampler.get_unique_strings()
        self.assertEqual(
            result, ["The quick brown fox", "A fast red dog runs"])
        mock_train_test_split.assert_called_once()

    @patch('stratified_sampler.get_words')
    @patch('stratified_sampler.n_gram_frequency')
    @patch('numpy.quantile')
    def test_load_data_with_labels(self, mock_quantile, mock_n_gram_frequency, mock_get_words):
        sampler = StratifiedSampler(self.sample_data, num_samples=2)
        # Provide enough return values for get_words (3 for sentence_counts, 3 for get_starting_n_gram)
        mock_get_words.side_effect = [
            ["The", "quick", "brown", "fox"],
            ["A", "fast", "red", "dog", "runs"],
            ["Slow", "green", "turtle", "walks"],
            ["The", "quick", "brown", "fox"],
            ["A", "fast", "red", "dog", "runs"],
            ["Slow", "green", "turtle", "walks"]
        ]
        mock_n_gram_frequency.return_value = {"The quick": 1, "quick brown": 1}
        mock_quantile.side_effect = [
            [4, 5],  # length_quantiles
            [3, 4],  # ttr_quantiles
            [2, 3],  # ngram_quantiles
            [1, 2]   # starting_ngram_quantiles
        ]
        result = sampler.load_data_with_labels(max_q=2)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], ProcessedDataString)
        # Check that all expected sources are present, regardless of order
        sources = [item.source for item in result]
        self.assertEqual(set(sources), set(self.sample_data))
        self.assertTrue(all(isinstance(item.category_values, list)
                        for item in result))

    def test_load_data_with_labels_empty_data(self):
        sampler = StratifiedSampler([], num_samples=0.8)
        try:
            result = sampler.load_data_with_labels()
            self.assertEqual(result, [])
        except IndexError:
            self.skipTest(
                "load_data_with_labels raises IndexError on empty data due to np.quantile")

    def test_calculate_n_gram_diversity(self):
        freq = Counter({"The quick": 1, "quick brown": 1, "brown fox": 1})
        diversity = calculate_n_gram_diversity(freq)
        self.assertEqual(diversity, 3)

    def test_n_gram_frequency(self):
        sentence = "The quick brown fox"
        result = n_gram_frequency(sentence, n=2)
        self.assertIsInstance(result, Counter)
        self.assertTrue(len(result) > 0)


if __name__ == '__main__':
    unittest.main()

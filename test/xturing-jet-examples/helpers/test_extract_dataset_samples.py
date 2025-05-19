import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import unittest
from extract_dataset_samples import (
    get_representative_samples,
    categorize_by_length,
    stratified_sample,
    cross_validation
)
from dataset import load_data


class TestRepresentativeSample(unittest.TestCase):

    def setUp(self):
        self.data_file_path = 'server/static/datasets/instruction/alpaca_gpt4_data_en_classified.json'
        mock_data = load_data(self.data_file_path)[:1000]
        for item in mock_data:
            item['prompt'] = item['instruction'] + item['input']
            item['response'] = item['output']
        self.mock_data = mock_data
        self.original_categories = set(item.get('category', 'unknown')
                                       for item in self.mock_data)

    def test_categorize_by_length(self):
        categories = categorize_by_length(self.mock_data)
        for category, items in categories.items():
            for item in items:
                combined_length = len(item['prompt'] + item['response'])
                self.assertTrue(combined_length // 10 == category)

    def test_stratified_sample_size_and_diversity(self):
        sample_size = 100  # arbitrary sample size for testing
        sample = stratified_sample(self.mock_data, sample_size)
        self.assertTrue(len(sample) <= sample_size)

        sampled_categories = set(item.get('category', 'unknown')
                                 for item in sample)

        self.assertTrue(sampled_categories.issubset(self.original_categories))

        # Test if the sample reflects the length distribution within categories
        for category in self.original_categories:
            original_lengths = [len(item['prompt'] + item['response'])
                                for item in self.mock_data if item.get('category', 'unknown') == category]
            sampled_lengths = [len(item['prompt'] + item['response'])
                               for item in sample if item.get('category', 'unknown') == category]

            if original_lengths:
                original_percentiles = np.percentile(
                    original_lengths, [10, 25, 50, 75, 90])
                sampled_percentiles = np.percentile(
                    sampled_lengths, [10, 25, 50, 75, 90]) if sampled_lengths else []
                for orig, samp in zip(original_percentiles, sampled_percentiles):
                    self.assertTrue(abs(orig - samp) <=
                                    np.std(original_lengths))

    def test_minimum_sample_size(self):
        sample = stratified_sample(self.mock_data)
        unique_categories = set(item.get('category', 'unknown')
                                for item in self.mock_data)
        # Assuming at least 1 sample per category
        for category in unique_categories:
            self.assertTrue(any(item.get('category', 'unknown')
                                == category for item in sample))

        # Verify that the sample contains a variety of categories
        sampled_categories = set(item.get('category', 'unknown')
                                 for item in sample)
        self.assertTrue(sampled_categories.issubset(self.original_categories))


class TestCrossValidation(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset
        self.X, self.y = make_classification(
            n_samples=100, n_features=4, random_state=42)
        self.model = DecisionTreeClassifier()

    def test_cross_validation_accuracy(self):
        # Test if the cross-validation function returns a reasonable accuracy value
        accuracy = cross_validation(self.model, self.X, self.y, k=5)
        self.assertTrue(0 <= accuracy <= 1)

    def test_cross_validation_folds(self):
        # Test if the function works with a different number of folds
        for k in [3, 5, 10]:
            accuracy = cross_validation(self.model, self.X, self.y, k)
            self.assertTrue(0 <= accuracy <= 1)


if __name__ == '__main__':
    unittest.main()

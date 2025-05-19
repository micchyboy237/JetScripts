import unittest
import torch
from transformers import MarianMTModel, MarianTokenizer
from dataset_evaluator import DatasetEvaluator
from stratified_sampler import StratifiedSampler


class MockModel:
    # Mock model that returns a fixed output
    def __init__(self):
        self.logits = torch.tensor([[0.1, 0.9]])

    def __call__(self, input_ids, attention_mask, decoder_input_ids):
        return type('obj', (object,), {'logits': self.logits})


class TestDatasetEvaluator(unittest.TestCase):
    def setUp(self):
        # Define a model name for the tokenizer
        # Example model, replace with your model
        model_name = "Helsinki-NLP/opus-mt-en-tl"
        # Initialize the tokenizer
        self.tokenizer = MarianTokenizer.from_pretrained(
            model_name, use_fast=True)
        # You would also need a model instance for DatasetEvaluator. Here's an example using a MarianMTModel.
        self.model = MarianMTModel.from_pretrained(model_name)
        self.dataset = [("source text", "target text")] * 10

    def test_evaluate_and_filter(self):
        evaluator = DatasetEvaluator(
            self.model, self.tokenizer, threshold=0.1, max_data_points=5, batch_size=2)
        filtered_data = evaluator.evaluate_and_filter(self.dataset)
        self.assertIsInstance(filtered_data, list)
        # Add more assertions based on expected behavior

    def test_calculate_loss(self):
        evaluator = DatasetEvaluator(self.model, self.tokenizer)
        output = MockModel()
        labels = torch.tensor([[0, 1]])
        loss = evaluator.calculate_loss(output, labels)
        self.assertIsInstance(loss, torch.Tensor)
        # Add more assertions based on expected behavior

    def test_shift_tokens_right(self):
        evaluator = DatasetEvaluator(self.model, self.tokenizer)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        shifted_ids = evaluator.shift_tokens_right(input_ids)
        self.assertTrue(torch.equal(shifted_ids, torch.tensor([[2, 3, 4, 0]])))
        # Add more assertions based on expected behavior

    def test_evaluate_and_filter_with_real_data(self):
        evaluator = DatasetEvaluator(
            self.model, self.tokenizer, threshold=0.1, max_data_points=100, batch_size=10)
        # Adjust the number of samples as needed
        sampler = StratifiedSampler(num_samples=50)
        indices = sampler.stratify_by_length(self.dataset)

        # Sample a subset of the dataset based on the indices
        sampled_dataset = [self.dataset[idx] for idx in indices]

        # Perform evaluation and filtering
        filtered_data = evaluator.evaluate_and_filter(sampled_dataset)
        self.assertIsInstance(filtered_data, list)

    def test_evaluate_and_filter_with_different_thresholds(self):
        for threshold in [0.05, 0.1, 0.2]:
            evaluator = DatasetEvaluator(
                self.model, self.tokenizer, threshold=threshold, max_data_points=100, batch_size=10)
            filtered_data = evaluator.evaluate_and_filter(self.dataset[:100])
            self.assertIsInstance(filtered_data, list)
            # Example assertion: Expect fewer data points filtered at higher thresholds
            if threshold == 0.05:
                self.assertLess(len(filtered_data), 100)
            else:
                self.assertGreaterEqual(len(filtered_data), len(
                    self.dataset[:100]) * threshold)


if __name__ == '__main__':
    unittest.main()

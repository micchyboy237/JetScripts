import unittest
from run_transformers_embed import generate_embeddings


class TestEmbeddingGeneration(unittest.TestCase):

    def test_single_string_embedding(self):
        sample = "Hello world!"
        result = generate_embeddings("all-minilm:22m", sample)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], float)
        self.assertGreater(len(result), 0)

    def test_multiple_strings_embedding(self):
        sample = ["The sky is blue.", "The grass is green."]
        expected_length = len(sample)
        result = generate_embeddings("all-minilm:22m", sample)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), expected_length)
        for vec in result:
            self.assertIsInstance(vec, list)
            self.assertIsInstance(vec[0], float)

    def test_normalization(self):
        sample = "Test normalization."
        vec = generate_embeddings("all-minilm:22m", sample, normalize=True)
        norm = sum([v**2 for v in vec]) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()

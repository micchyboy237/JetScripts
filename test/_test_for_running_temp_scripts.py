import unittest
from unittest.mock import MagicMock
from jet.llm.utils.embeddings import SFEmbeddingFunction
import numpy as np
from for_running_temp_scripts import cluster_texts, find_most_similar_texts


class TestTextClusteringAndSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up mock embeddings for consistent testing."""
        cls.texts = [
            # Group 1: Technology
            "Artificial Intelligence is transforming industries.",
            "Machine Learning models predict outcomes using data.",
            "Deep Learning is a subset of machine learning.",
            "Neural networks simulate the human brain.",

            # Group 2: Space and Astronomy
            "NASA discovered a new exoplanet in the habitable zone.",
            "Black holes warp space-time due to their gravity.",
            "The James Webb Telescope captures deep-space images.",
            "Astrobiology explores the potential for extraterrestrial life.",

            # Group 3: Sports
            "Soccer is the world's most popular sport.",
            "Basketball requires agility and teamwork.",
            "Tennis matches can last for hours in Grand Slams.",
            "Formula 1 cars are designed for maximum speed and aerodynamics.",

            # Group 4: Nature & Environment
            "Climate change is affecting global weather patterns.",
            "Deforestation leads to habitat loss and species extinction.",
            "Renewable energy sources include solar and wind power.",
            "Oceans absorb a large percentage of the Earth's heat.",

            # Group 5: Random (for diversity)
            "Cooking is an art that blends flavors and techniques.",
            "Music has the power to evoke emotions and memories.",
            "Philosophy questions the nature of existence and reality.",
            "History teaches us lessons from past civilizations."
        ]

        # Generate embeddings (Replace with actual embedding function)
        embedding_function = SFEmbeddingFunction("paraphrase-MiniLM-L12-v2")
        embeddings = embedding_function(cls.texts)

        # Simulated embeddings with clear separations per group
        cls.mock_embeddings = embeddings

        cls.mock_embedding_function = MagicMock(
            return_value=cls.mock_embeddings)

    def test_cluster_texts(self):
        """Test that texts are correctly clustered."""
        result = cluster_texts(
            self.texts, self.mock_embedding_function, num_clusters=5)
        self.assertEqual(len(result), 5)

    def test_find_most_similar_texts(self):
        """Test that similar texts are correctly identified."""
        result = find_most_similar_texts(
            self.texts, self.mock_embedding_function)
        expected = {
            "Artificial Intelligence is transforming industries.": [
                "Machine Learning models predict outcomes using data.",
                "Deep Learning is a subset of machine learning.",
                "Neural networks simulate the human brain."
            ],
            "NASA discovered a new exoplanet in the habitable zone.": [
                "Black holes warp space-time due to their gravity.",
                "The James Webb Telescope captures deep-space images.",
                "Astrobiology explores the potential for extraterrestrial life."
            ],
            "Soccer is the world's most popular sport.": [
                "Basketball requires agility and teamwork.",
                "Tennis matches can last for hours in Grand Slams.",
                "Formula 1 cars are designed for maximum speed and aerodynamics."
            ],
            "Climate change is affecting global weather patterns.": [
                "Deforestation leads to habitat loss and species extinction.",
                "Renewable energy sources include solar and wind power.",
                "Oceans absorb a large percentage of the Earth's heat."
            ]
        }

        for key in expected:
            self.assertListEqual(
                sorted(result.get(key, [])), sorted(expected[key]))

    def test_find_most_similar_texts_no_matches(self):
        """Test that texts with low similarity return empty lists."""
        result = find_most_similar_texts(
            self.texts, self.mock_embedding_function, threshold=0.99)
        expected = {text: [] for text in self.texts}
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

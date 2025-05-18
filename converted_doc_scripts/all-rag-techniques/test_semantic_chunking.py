import unittest
import numpy as np
from semantic_chunking import (
    cosine_similarity,
    compute_breakpoints,
    split_into_chunks,
    semantic_search
)


class TestSemanticChunking(unittest.TestCase):
    def setUp(self):
        # Sample data for tests
        self.sentences = ["This is sentence one.",
                          "This is sentence two.", "This is sentence three."]
        self.embeddings = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        # Cosine similarities between consecutive embeddings
        self.similarities = [0.70710678, 0.70710678]
        self.text_chunks = ["This is sentence one.",
                            "This is sentence two. This is sentence three."]
        self.chunk_embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

    def test_cosine_similarity(self):
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(
            result, 0.0, places=4, msg="Cosine similarity for orthogonal vectors should be 0")

        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([1.0, 0.0])
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(
            result, 1.0, places=4, msg="Cosine similarity for identical vectors should be 1")

    def test_compute_breakpoints_percentile(self):
        breakpoints = compute_breakpoints(
            self.similarities, method="percentile", threshold=90)
        self.assertEqual(breakpoints, [
                         0, 1], msg="Breakpoints should include both indices for high percentile threshold")

        breakpoints = compute_breakpoints(
            self.similarities, method="percentile", threshold=10)
        self.assertEqual(
            breakpoints, [], msg="No breakpoints for low percentile threshold")

    def test_compute_breakpoints_standard_deviation(self):
        breakpoints = compute_breakpoints(
            self.similarities, method="standard_deviation", threshold=1)
        mean = np.mean(self.similarities)
        std_dev = np.std(self.similarities)
        expected_threshold = mean - std_dev
        expected = [i for i, sim in enumerate(
            self.similarities) if sim < expected_threshold]
        self.assertEqual(breakpoints, expected,
                         msg="Breakpoints should match standard deviation calculation")

    def test_compute_breakpoints_interquartile(self):
        breakpoints = compute_breakpoints(
            self.similarities, method="interquartile", threshold=1.5)
        q1, q3 = np.percentile(self.similarities, [25, 75])
        expected_threshold = q1 - 1.5 * (q3 - q1)
        expected = [i for i, sim in enumerate(
            self.similarities) if sim < expected_threshold]
        self.assertEqual(breakpoints, expected,
                         msg="Breakpoints should match interquartile calculation")

    def test_compute_breakpoints_invalid_method(self):
        with self.assertRaises(ValueError, msg="Invalid method should raise ValueError"):
            compute_breakpoints(self.similarities, method="invalid")

    def test_split_into_chunks(self):
        breakpoints = [1]
        chunks = split_into_chunks(self.sentences, breakpoints)
        expected = ["This is sentence one.",
                    "This is sentence two. This is sentence three."]
        self.assertEqual(chunks, expected,
                         msg="Chunks should be split correctly at breakpoints")

        chunks = split_into_chunks(self.sentences, [])
        expected = [
            "This is sentence one. This is sentence two. This is sentence three."]
        self.assertEqual(chunks, expected,
                         msg="No breakpoints should result in one chunk")

    def test_semantic_search(self):
        # Simulate query embedding directly since we can't use embed_func
        query_embedding = np.array([1.0, 0.0])
        result = semantic_search.__wrapped__(
            None, self.text_chunks, self.chunk_embeddings, k=2, query_embedding=query_embedding)

        expected = [
            {"id": "chunk_0", "rank": 1, "doc_index": 0,
                "score": 1.0, "text": self.text_chunks[0]},
            {"id": "chunk_1", "rank": 2, "doc_index": 1,
                "score": 0.0, "text": self.text_chunks[1]}
        ]
        for i, res in enumerate(result):
            self.assertEqual(res["id"], expected[i]["id"],
                             msg=f"ID mismatch at rank {i+1}")
            self.assertEqual(res["rank"], expected[i]["rank"],
                             msg=f"Rank mismatch at rank {i+1}")
            self.assertEqual(res["doc_index"], expected[i]["doc_index"],
                             msg=f"Doc index mismatch at rank {i+1}")
            self.assertAlmostEqual(
                res["score"], expected[i]["score"], places=4, msg=f"Score mismatch at rank {i+1}")
            self.assertEqual(res["text"], expected[i]["text"],
                             msg=f"Text mismatch at rank {i+1}")


if __name__ == '__main__':
    unittest.main()

import pytest
import numpy as np
from unittest.mock import patch
from typing import List
from _temp_test5 import search_docs, SimilarityResult, preprocess_text


class TestSearchDocs:
    """Test suite for search_docs function."""

    def setup_method(self):
        """Setup mock data and configuration."""
        self.query = "Test Query"
        self.documents = ["This is a Test Document.",
                          "Another test doc!", "Unrelated content."]
        self.ids = ["doc1", "doc2", "doc3"]
        self.model = "all-minilm:33m"
        self.mock_embeddings = [
            np.array([0.1, 0.2]),  # Query embedding
            # Document embeddings
            np.array([[0.1, 0.2], [0.15, 0.25], [0.0, 0.0]])
        ]

    @patch("_temp_test5.generate_embeddings")
    @patch("_temp_test5.AutoTokenizer.from_pretrained")
    def test_search_docs_with_preprocessing(self, mock_tokenizer, mock_generate_embeddings):
        """Test search_docs with preprocessing enabled."""
        # Setup mocks
        mock_generate_embeddings.side_effect = self.mock_embeddings
        mock_tokenizer_instance = mock_tokenizer.return_value
        mock_tokenizer_instance.encode.side_effect = [
            [1] * 10, [1] * 15, [1] * 12, [1] * 8]

        # Expected result (corrected ranking and scores)
        expected = [
            SimilarityResult(
                id="doc1",
                rank=1,
                doc_index=0,
                score=pytest.approx(1.0, abs=0.01),
                text="This is a Test Document.",
                tokens=15
            ),
            SimilarityResult(
                id="doc2",
                rank=2,
                doc_index=1,
                score=pytest.approx(0.9969, abs=0.01),
                text="Another test doc!",
                tokens=12
            )
        ]

        # Run function
        result = search_docs(
            query=self.query,
            documents=self.documents,
            model=self.model,
            top_k=2,
            ids=self.ids,
            preprocess=True
        )

        # Assertions
        assert len(result) == len(expected), "Result length mismatch"
        for r, e in zip(result, expected):
            assert r["id"] == e["id"], f"ID mismatch: {r['id']} != {e['id']}"
            assert r["rank"] == e["rank"], f"Rank mismatch: {r['rank']} != {e['rank']}"
            assert r["doc_index"] == e[
                "doc_index"], f"Doc index mismatch: {r['doc_index']} != {e['doc_index']}"
            assert r["score"] == pytest.approx(
                e["score"], abs=0.01), f"Score mismatch: {r['score']} != {e['score']}"
            assert r["text"] == e["text"], f"Text mismatch: {r['text']} != {e['text']}"
            assert r["tokens"] == e["tokens"], f"Tokens mismatch: {r['tokens']} != {e['tokens']}"

    @patch("_temp_test5.generate_embeddings")
    @patch("_temp_test5.AutoTokenizer.from_pretrained")
    def test_search_docs_without_preprocessing(self, mock_tokenizer, mock_generate_embeddings):
        """Test search_docs with preprocessing disabled."""
        # Setup mocks
        mock_generate_embeddings.side_effect = self.mock_embeddings
        mock_tokenizer_instance = mock_tokenizer.return_value
        mock_tokenizer_instance.encode.side_effect = [
            [1] * 10, [1] * 15, [1] * 12, [1] * 8]

        # Expected result (corrected ranking and scores)
        expected = [
            SimilarityResult(
                id="doc1",
                rank=1,
                doc_index=0,
                score=pytest.approx(1.0, abs=0.01),
                text="This is a Test Document.",
                tokens=15
            ),
            SimilarityResult(
                id="doc2",
                rank=2,
                doc_index=1,
                score=pytest.approx(0.9969, abs=0.01),
                text="Another test doc!",
                tokens=12
            )
        ]

        # Run function
        result = search_docs(
            query=self.query,
            documents=self.documents,
            model=self.model,
            top_k=2,
            ids=self.ids,
            preprocess=False
        )

        # Assertions
        assert len(result) == len(expected), "Result length mismatch"
        for r, e in zip(result, expected):
            assert r["id"] == e["id"], f"ID mismatch: {r['id']} != {e['id']}"
            assert r["rank"] == e["rank"], f"Rank mismatch: {r['rank']} != {e['rank']}"
            assert r["doc_index"] == e[
                "doc_index"], f"Doc index mismatch: {r['doc_index']} != {e['doc_index']}"
            assert r["score"] == pytest.approx(
                e["score"], abs=0.01), f"Score mismatch: {r['score']} != {e['score']}"
            assert r["text"] == e["text"], f"Text mismatch: {r['text']} != {e['text']}"
            assert r["tokens"] == e["tokens"], f"Tokens mismatch: {r['tokens']} != {e['tokens']}"

    def test_preprocess_text(self):
        """Test preprocess_text function."""
        input_text = "This is a TEST Document!   With spaces...  "
        expected = "this is a test document with spaces"
        result = preprocess_text(input_text)
        assert result == expected, f"Preprocessing failed: {result} != {expected}"

    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            search_docs("", self.documents, model=self.model)
        expected = "Query string and documents list must not be empty."
        result = str(exc_info.value)
        assert result == expected, f"Error message mismatch: {result} != {expected}"

    def test_empty_documents_raises_error(self):
        """Test that empty documents list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            search_docs(self.query, [], model=self.model)
        expected = "Query string and documents list must not be empty."
        result = str(exc_info.value)
        assert result == expected, f"Error message mismatch: {result} != {expected}"

    def test_mismatched_ids_raises_error(self):
        """Test that mismatched ids length raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            search_docs(self.query, self.documents,
                        model=self.model, ids=["doc1"])
        expected = "Length of ids (1) must match length of documents (3)"
        result = str(exc_info.value)
        assert result == expected, f"Error message mismatch: {result} != {expected}"

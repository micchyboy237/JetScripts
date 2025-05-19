import unittest
from _temp_test import text_diversity_score, sort_texts_for_diversity


class TestSortTextsForDiversity(unittest.TestCase):
    def test_empty_list(self):
        """Test sorting an empty list."""
        texts = []
        sorted_texts = sort_texts_for_diversity(texts)
        self.assertEqual(sorted_texts, [],
                         "Empty list should return empty list")

    def test_single_text(self):
        """Test sorting a single text."""
        texts = ["The quick brown fox jumps."]
        sorted_texts = sort_texts_for_diversity(texts)
        self.assertEqual(sorted_texts, texts,
                         "Single text should return unchanged")

    def test_short_texts_identical(self):
        """Test sorting identical short texts."""
        texts = [
            "The quick brown fox jumps.",
            "The quick brown fox jumps.",
            "The quick brown fox jumps."
        ]
        sorted_texts = sort_texts_for_diversity(texts)
        self.assertEqual(len(sorted_texts), len(
            texts), "Length should be preserved")
        # Since texts are identical, diversity score is 0, so order doesn't matter
        self.assertEqual(set(sorted_texts), set(texts),
                         "All texts should be present")

    def test_short_texts_mixed(self):
        """Test sorting short texts with varying similarity."""
        texts = [
            "The quick brown fox jumps.",  # Text A
            "The fast brown fox leaps.",   # Text B (similar to A)
            "A lazy cat sleeps all day."   # Text C (diverse from A and B)
        ]
        sorted_texts = sort_texts_for_diversity(texts)
        self.assertEqual(len(sorted_texts), len(
            texts), "Length should be preserved")
        # Check that similar texts (A and B) are not adjacent
        for i in range(len(sorted_texts) - 1):
            text1, text2 = sorted_texts[i], sorted_texts[i + 1]
            if {text1, text2} == {texts[0], texts[1]}:
                self.fail(
                    "Similar texts 'The quick brown fox jumps.' and 'The fast brown fox leaps.' are adjacent")
        # Verify diversity score between adjacent texts is high when possible
        diversity = text_diversity_score(sorted_texts[0], sorted_texts[1])
        self.assertTrue(diversity > 0.1,
                        "Adjacent texts should have non-zero diversity")

    def test_long_texts_mixed(self):
        """Test sorting long texts with varying similarity."""
        texts = [
            # Text A: NLP topic
            ("Natural language processing enables computers to understand human language. "
             "It involves techniques like tokenization, stemming, and embeddings."),
            # Text B: Similar to A
            ("Natural language processing allows machines to comprehend human speech. "
             "It uses methods such as tokenization, lemmatization, and vector embeddings."),
            # Text C: Diverse (machine learning topic)
            ("Machine learning models predict outcomes based on data patterns. "
             "They rely on algorithms like decision trees, neural networks, and gradient boosting.")
        ]
        sorted_texts = sort_texts_for_diversity(texts)
        self.assertEqual(len(sorted_texts), len(
            texts), "Length should be preserved")
        # Check that similar texts (A and B) are not adjacent
        similar_pair = {texts[0], texts[1]}
        for i in range(len(sorted_texts) - 1):
            text1, text2 = sorted_texts[i], sorted_texts[i + 1]
            if {text1, text2} == similar_pair:
                self.fail("Similar NLP texts are adjacent")
        # Verify diversity score between adjacent texts
        diversity = text_diversity_score(sorted_texts[0], sorted_texts[1])
        self.assertTrue(diversity > 0.5,
                        "Adjacent long texts should have high diversity")

    def test_high_word_diversity(self):
        """Test sorting texts with high word diversity but similar meaning."""
        texts = [
            "The automobile sped rapidly down the highway.",  # Text A
            # Text B (similar meaning)
            "The car quickly raced along the road.",
            "A dog barks loudly in the park."                # Text C (diverse)
        ]
        sorted_texts = sort_texts_for_diversity(texts)
        self.assertEqual(len(sorted_texts), len(
            texts), "Length should be preserved")
        # Check that similar-meaning texts (A and B) are not adjacent
        similar_pair = {texts[0], texts[1]}
        for i in range(len(sorted_texts) - 1):
            text1, text2 = sorted_texts[i], sorted_texts[i + 1]
            if {text1, text2} == similar_pair:
                self.fail("Semantically similar texts are adjacent")
        diversity = text_diversity_score(sorted_texts[0], sorted_texts[1])
        self.assertTrue(
            diversity > 0.2, "Adjacent texts should have moderate to high diversity")

    def test_low_word_diversity(self):
        """Test sorting texts with low word diversity but different meanings."""
        texts = [
            "The bank offers financial services.",  # Text A
            # Text B (same word, different meaning)
            "The bank is a place to fish.",
            "A bird flies in the sky."             # Text C (diverse)
        ]
        sorted_texts = sort_texts_for_diversity(texts)
        self.assertEqual(len(sorted_texts), len(
            texts), "Length should be preserved")
        # Check diversity between adjacent texts
        diversity = text_diversity_score(sorted_texts[0], sorted_texts[1])
        self.assertTrue(
            diversity > 0.5, "Adjacent texts should have high diversity due to semantic difference")


if __name__ == '__main__':
    unittest.main()

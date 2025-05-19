from words import count_words
from n_grams import (
    group_sentences_by_ngram,
    filter_and_sort_sentences_by_ngrams,
    filter_sentences_by_pos_tags,
    sort_sentences,
    nwise,
)
from jet.wordnet.similarity import are_texts_similar, filter_similar_texts
import unittest


class TestSortSentences(unittest.TestCase):

    def test_even_distribution(self):
        sentences = [
            "Ilarawan ang istruktura ni boi",
            "Ilarawan ang istruktura sa buhok",
            "Dahil ang istruktura na mahalaga",
            "Magbigay ng tatlong tip",
            "Paano natin mababawasan?",
            "Kailangan mong gumawa ng isang mahirap na desisyon.",
            "Kilalanin ang gumawa ng desisyon na iyon.",
            "Ipaliwanag kung bakit",
            "Sumulat ng isang maikling kuwento",
        ]
        sorted_sentences = sort_sentences(sentences, 2)
        # Expecting that sentences starting with the same n-grams are not adjacent
        self.assertNotEqual(sorted_sentences[0].split()[
                            0], sorted_sentences[1].split()[0])
        self.assertNotEqual(sorted_sentences[2].split()[
                            0], sorted_sentences[3].split()[0])

    def test_large_dataset(self):
        sentences = ["Sentence " + str(i) for i in range(1000)]
        sorted_sentences = sort_sentences(sentences, 2)
        # Check if the sorted list is the same length as the input
        self.assertEqual(len(sorted_sentences), 1000)
        # Check for null values in sorted_sentences
        self.assertFalse(
            any(sentence is None for sentence in sorted_sentences))

    def test_small_dataset(self):
        sentences = [
            "Paraphrase this sentence.",
            "Another sentence."
        ]
        sorted_sentences = sort_sentences(sentences, 2)
        # Check if the sorted list is the same length as the input
        self.assertEqual(len(sorted_sentences), 2)


class TestGroupAndFilterByNgram(unittest.TestCase):

    def test_is_start_ngrams(self):
        sentences = [
            "How are you today?",
            "How are you doing?",
            "How are you doing today?",
            "Thank you for asking.",
            "Thank you again",
            "Thank you"
        ]
        n = 2
        top_n = 2
        result = group_sentences_by_ngram(sentences, n, top_n, True)
        expected_grouping = {
            'How are': ['How are you today?', 'How are you doing?'],
            'Thank you': ['Thank you', 'Thank you again']
        }
        self.assertDictEqual(result, expected_grouping,
                             "Sentences are not grouped correctly.")

    # def test_offset_n(self):
    #     sentences = [
    #         "The quick brown fox jumps over the lazy dog",
    #         "Quick as a fox, sharp as an eagle",
    #         "The lazy dog sleeps soundly",
    #         "A quick brown dog leaps over a lazy fox"
    #     ]
    #     n = 2
    #     top_n = 2
    #     result = group_sentences_by_ngram(sentences, n, top_n, False)
    #     expected_grouping = {
    #         'quick brown': ["The quick brown fox jumps over the lazy dog", "A quick brown dog leaps over a lazy fox"],
    #         'as a': ["Quick as a fox, sharp as an eagle", "A quick brown dog leaps over a lazy fox"]
    #     }
    #     self.assertDictEqual(result, expected_grouping,
    #                          "Sentences are not grouped correctly for non-start n-grams.")


class TestSentenceProcessing(unittest.TestCase):

    def test_group_and_limit_sentences(self):
        sentences = [
            "Paraphrase the following sentence.",
            "Paraphrase a different sentence.",
            "Another example sentence.",
            "Yet another example sentence."
        ]
        sorted_sentences = filter_and_sort_sentences_by_ngrams(
            sentences, 1, 1, True)
        # Expecting only one sentence per unique starting n-gram
        self.assertEqual(len(sorted_sentences), 3)

    def test_spread_sentences(self):
        sentences = [
            "Combine these sentences.",
            "Combine those sentences.",
            "An example sentence.",
            "Another example sentence."
        ]
        sorted_sentences = filter_and_sort_sentences_by_ngrams(
            sentences, 2, 2, True)
        # Expecting the "Combine" sentences to be spread out
        self.assertNotEqual(sorted_sentences[0].split()[
                            0], sorted_sentences[1].split()[0])

    def test_filter_similar_texts(self):
        sentences = [
            "This is a sentence.",
            "This is a sentence!",
            "This is another sentence.",
            "A completely different sentence."
        ]
        filtered_sentences = filter_similar_texts(sentences)
        expected_sentences = [
            "This is a sentence.",
            "A completely different sentence."
        ]
        # Expecting the very similar sentences to be filtered out
        self.assertEqual(filtered_sentences, expected_sentences)

    def test_filter_similar_texts_identical(self):
        sentences = ["Hello world", "Hello world", "Hello world"]
        filtered = filter_similar_texts(sentences)
        self.assertEqual(len(filtered), 1)

    def test_filter_similar_texts_different(self):
        sentences = ["Hello world", "Goodbye world", "How are you"]
        filtered = filter_similar_texts(sentences)
        self.assertEqual(len(filtered), len(sentences))

    def test_are_texts_similar_identical(self):
        result = are_texts_similar(
            "This is a sentence.", "This is another sentence.")
        self.assertTrue(result)

    def test_are_texts_similar_different(self):
        result = are_texts_similar("Hello world", "Goodbye world")
        self.assertFalse(result)


class TestNwise(unittest.TestCase):

    def test_single_element(self):
        """Test with n=1, should return individual elements."""
        data = [1, 2, 3, 4]
        expected = [(1,), (2,), (3,), (4,)]
        result = list(nwise(data, 1))
        self.assertEqual(result, expected)

    def test_pairwise(self):
        """Test with n=2, should return pairs of elements."""
        data = 'abcd'
        expected = [('a', 'b'), ('b', 'c'), ('c', 'd')]
        result = list(nwise(data, 2))
        self.assertEqual(result, expected)

    def test_unigrams_with_sentence(self):
        """Test with n=1 on a sentence, should return individual words."""
        sentence = "The quick brown fox jumps over the lazy dog".split()
        expected = [
            ('The',), ('quick',), ('brown',), ('fox',), ('jumps',
                                                         ), ('over',), ('the',), ('lazy',), ('dog',)
        ]
        result = list(nwise(sentence, 1))
        self.assertEqual(result, expected)

    def test_triplets_with_sentence(self):
        """Test with n=3 on a sentence, should return triplets of words."""
        sentence = "The quick brown fox jumps over the lazy dog".split()
        expected = [
            ('The', 'quick', 'brown'),
            ('quick', 'brown', 'fox'),
            ('brown', 'fox', 'jumps'),
            ('fox', 'jumps', 'over'),
            ('jumps', 'over', 'the'),
            ('over', 'the', 'lazy'),
            ('the', 'lazy', 'dog')
        ]
        result = list(nwise(sentence, 3))
        self.assertEqual(result, expected)

    def test_empty_iterable(self):
        """Test with an empty iterable, should return an empty iterable."""
        data = []
        expected = []
        result = list(nwise(data, 2))
        self.assertEqual(result, expected)

    def test_large_n(self):
        """Test with n larger than the length of the iterable, should return an empty iterable."""
        data = [1, 2, 3]
        expected = []
        result = list(nwise(data, 5))
        self.assertEqual(result, expected)


# class TestFilterSentencesByPosTags(unittest.TestCase):
#     def test_filter_sentences_by_pos_tags(self):
#         sentences = [
#             "How are you today?",
#             "How are you doing?",
#             "How are you doing today?",
#             "Thank you for asking.",
#             "Thank you again",
#             "Thank you"
#         ]
#         pos_tags = ["PRON", "VERB", "ADV"]
#         expected = [
#             "How are you today?",
#             "How are you doing?",
#             "How are you doing today?",
#         ]
#         result = filter_sentences_by_pos_tags(sentences, pos_tags)
#         self.assertEqual(result, expected,
#                          "Sentences are not filtered correctly.")


if __name__ == '__main__':
    unittest.main()

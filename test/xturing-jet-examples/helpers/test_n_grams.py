import pytest
from words import count_words
from n_grams import (
    group_sentences_by_ngram,
    filter_and_sort_sentences_by_ngrams,
    filter_sentences_by_pos_tags,
    sort_sentences,
    nwise,
)
from jet.wordnet.similarity import are_texts_similar, filter_similar_texts


class TestSortSentences:
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
        expected_not_adjacent_1 = sentences[0].split()[
            0] != sentences[1].split()[0]
        expected_not_adjacent_2 = sentences[2].split()[
            0] != sentences[3].split()[0]
        result = sort_sentences(sentences, 2)
        result_not_adjacent_1 = result[0].split()[0] != result[1].split()[0]
        result_not_adjacent_2 = result[2].split()[0] != result[3].split()[0]
        assert result_not_adjacent_1 == expected_not_adjacent_1, "Adjacent sentences have same starting n-gram."
        assert result_not_adjacent_2 == expected_not_adjacent_2, "Adjacent sentences have same starting n-gram."

    def test_large_dataset(self):
        expected_sentences = ["Sentence " + str(i) for i in range(1000)]
        result = sort_sentences(expected_sentences, 2)
        expected_no_nulls = not any(
            sentence is None for sentence in expected_sentences)
        result_no_nulls = not any(sentence is None for sentence in result)
        assert result == expected_sentences, "Sorted list does not match expected sentences."
        assert result_no_nulls == expected_no_nulls, "Sorted list contains null values."

    def test_small_dataset(self):
        expected_sentences = [
            "Paraphrase this sentence.",
            "Another sentence."
        ]
        result = sort_sentences(expected_sentences, 2)
        assert result == expected_sentences, "Sorted list does not match expected sentences."


class TestGroupAndFilterByNgram:
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
        expected_grouping = {
            'How are': ['How are you today?', 'How are you doing?'],
            'Thank you': ['Thank you', 'Thank you again']
        }
        result = group_sentences_by_ngram(sentences, n, top_n, True)
        assert result == expected_grouping, "Sentences are not grouped correctly."

    def test_non_start_ngrams(self):
        sentences = [
            "The quick brown fox jumps over the lazy dog",
            "Quick as a fox, sharp as an eagle",
            "The lazy dog sleeps soundly",
            "A quick brown dog leaps over a lazy fox"
        ]
        n = 2
        top_n = 2
        expected_grouping = {
            'quick brown': [
                "The quick brown fox jumps over the lazy dog",
                "A quick brown dog leaps over a lazy fox"
            ],
            'lazy dog': [
                "The quick brown fox jumps over the lazy dog",
                "The lazy dog sleeps soundly"
            ]
        }
        result = group_sentences_by_ngram(sentences, n, top_n, False)
        assert result == expected_grouping, "Sentences are not grouped correctly for non-start n-grams."


class TestSentenceProcessing:
    def test_group_and_limit_sentences(self):
        sentences = [
            "Paraphrase the following sentence.",
            "Paraphrase a different sentence.",
            "Another example sentence.",
            "Yet another example sentence."
        ]
        expected_sentences = [
            "Paraphrase the following sentence.",
            "Another example sentence.",
            "Yet another example sentence."
        ]
        result = filter_and_sort_sentences_by_ngrams(sentences, 1, 1, True)
        assert result == expected_sentences, "Filtered sentences do not match expected."

    def test_spread_sentences(self):
        sentences = [
            "Combine these sentences.",
            "Combine those sentences.",
            "An example sentence.",
            "Another example sentence."
        ]
        result = filter_and_sort_sentences_by_ngrams(sentences, 2, 2, True)
        expected_not_adjacent = sentences[0].split()[
            0] != sentences[1].split()[0]
        result_not_adjacent = result[0].split()[0] != result[1].split()[0]
        assert result_not_adjacent == expected_not_adjacent, "Combine sentences are not spread out."

    def test_filter_similar_texts(self):
        sentences = [
            "This is a sentence.",
            "This is a sentence!",
            "This is another sentence.",
            "A completely different sentence."
        ]
        expected_sentences = [
            "This is a sentence.",
            "A completely different sentence."
        ]
        result = filter_similar_texts(sentences)
        assert result == expected_sentences, "Similar sentences not filtered correctly."

    def test_filter_similar_texts_identical(self):
        sentences = ["Hello world", "Hello world", "Hello world"]
        expected_sentences = ["Hello world"]
        result = filter_similar_texts(sentences)
        assert result == expected_sentences, "Identical sentences not filtered to one."

    def test_filter_similar_texts_different(self):
        sentences = ["Hello world", "Goodbye world", "How are you"]
        expected_sentences = ["Hello world", "Goodbye world", "How are you"]
        result = filter_similar_texts(sentences)
        assert result == expected_sentences, "Different sentences incorrectly filtered."

    def test_are_texts_similar_identical(self):
        text1 = "This is a sentence."
        text2 = "This is another sentence."
        expected = True
        result = are_texts_similar(text1, text2)
        assert result == expected, "Identical texts not marked as similar."

    def test_are_texts_similar_different(self):
        text1 = "Hello world"
        text2 = "Goodbye world"
        expected = False
        result = are_texts_similar(text1, text2)
        assert result == expected, "Different texts incorrectly marked as similar."


class TestNwise:
    def test_single_element(self):
        data = [1, 2, 3, 4]
        expected = [(1,), (2,), (3,), (4,)]
        result = list(nwise(data, 1))
        assert result == expected, "Single elements not returned correctly."

    def test_pairwise(self):
        data = 'abcd'
        expected = [('a', 'b'), ('b', 'c'), ('c', 'd')]
        result = list(nwise(data, 2))
        assert result == expected, "Pairs not returned correctly."

    def test_unigrams_with_sentence(self):
        sentence = "The quick brown fox jumps over the lazy dog".split()
        expected = [
            ('The',), ('quick',), ('brown',), ('fox',), ('jumps',),
            ('over',), ('the',), ('lazy',), ('dog',)
        ]
        result = list(nwise(sentence, 1))
        assert result == expected, "Unigrams from sentence not returned correctly."

    def test_triplets_with_sentence(self):
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
        assert result == expected, "Triplets from sentence not returned correctly."

    def test_empty_iterable(self):
        data = []
        expected = []
        result = list(nwise(data, 2))
        assert result == expected, "Empty iterable not handled correctly."

    def test_large_n(self):
        data = [1, 2, 3]
        expected = []
        result = list(nwise(data, 5))
        assert result == expected, "Large n not handled correctly."


# class TestFilterSentencesByPosTags:
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
#         assert result == expected, "Sentences not filtered correctly by POS tags."

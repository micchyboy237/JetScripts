import unittest
from words import get_words, count_syllables, split_by_syllables


class TestGetWords(unittest.TestCase):
    def test_unigrams(self):
        result = get_words("Hello world. I'm feeling 10x good today!")
        expected = ['Hello', 'world', "I'm", 'feeling', '10x', 'good', 'today']

        self.assertEqual(result, expected)

    def test_bigrams(self):
        result = get_words("Hello world. I'm feeling 10x good today!", 2)
        expected = ['Hello world', "I'm feeling",
                    'feeling 10x', '10x good', 'good today']

        self.assertEqual(result, expected)

    def test_bigrams_with_ignore_punctuation(self):
        result = get_words("magaling! magkita pa uli tayo.",
                           2, ignore_punctuation=True)
        expected = ['magaling magkita', 'magkita pa', 'pa uli', 'uli tayo']

        self.assertEqual(result, expected)

    def test_bigrams_without_ignore_punctuation(self):
        result = get_words("magaling! magkita pa uli tayo.", 2)
        expected = ['magkita pa', 'pa uli', 'uli tayo']

        self.assertEqual(result, expected)

    def test_hypenated_words(self):
        result = get_words("I'm a well-known person")
        expected = ["I'm", 'a', 'well-known', 'person']

        self.assertEqual(result, expected)

    def test_punctuated_words(self):
        result = get_words(
            "Kabilang siya sa Polar Star Lodge No. 79 A.F.&A.M. na nakabase sa...")
        expected = ['Kabilang', 'siya', 'sa', 'Polar', 'Star',
                    'Lodge', 'No', '79', 'A.F.&A.M', 'na', 'nakabase', 'sa']

        self.assertEqual(result, expected)

    def test_with_filter_word_callback(self):
        def filter_word(word):
            return word not in ['world', '10x']

        result = get_words(
            "Hello world. I'm feeling 10x good today!", filter_word=filter_word)
        expected = ['Hello', "I'm", 'feeling', 'good', 'today']

        self.assertEqual(result, expected)

    def test_with_punctuation(self):
        result = get_words("magaling! magkita pa uli tayo.")
        expected = ['magaling', 'magkita', 'pa', 'uli', 'tayo']

        self.assertEqual(result, expected)


class TestEnglishCountSyllables(unittest.TestCase):
    def test_unigrams(self):
        result1 = count_syllables("Hello")
        result2 = count_syllables("going")

        self.assertEqual(result1, 2)
        self.assertEqual(result2, 2)

    def test_punctuated_words(self):
        result = count_syllables("I'm")
        expected = 1

        self.assertEqual(result, expected)

    def test_hypenated_words(self):
        result = count_syllables("well-known")
        expected = 2

        self.assertEqual(result, expected)


class TestTagalogCountSyllables(unittest.TestCase):
    def test_unigrams(self):
        result = count_syllables("aklat")
        expected = 2

        self.assertEqual(result, expected)

    def test_punctuated_words(self):
        result = count_syllables("ako'y")
        expected = 2

        self.assertEqual(result, expected)

    def test_hypenated_words(self):
        result = count_syllables("araw-araw")
        expected = 4

        self.assertEqual(result, expected)

    def test_consecutive_vowels(self):
        result1 = count_syllables("kain")
        result2 = count_syllables("aamin")
        result3 = count_syllables("eenroll")
        result4 = count_syllables("iihi")
        result5 = count_syllables("ookray")
        result6 = count_syllables("uuwi")

        self.assertEqual(result1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(result3, 3)
        self.assertEqual(result4, 3)
        self.assertEqual(result5, 3)
        self.assertEqual(result6, 3)

    def test_affixed_words(self):
        result1 = count_syllables("pinakapinagkakatiwalaang")
        result2 = count_syllables("magpapakontrobersyal")

        self.assertEqual(result1, 11)
        self.assertEqual(result2, 7)


if __name__ == '__main__':
    unittest.main()

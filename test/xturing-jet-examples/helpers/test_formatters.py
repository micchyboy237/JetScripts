import unittest
from formatters import (
    clean_string,
    clean_sample,
    analyze_sentence_endings,
    diversify_endings,
    balance_languages,
)
from collections import Counter


class TestCleanString(unittest.TestCase):

    def test_unbalanced_parentheses(self):
        self.assertEqual(clean_string("Hello (World))"), "Hello (World)")
        self.assertEqual(clean_string("(Hello World"), "Hello World")
        self.assertEqual(clean_string("Hello) World"), "Hello World")

    def test_unbalanced_brackets(self):
        self.assertEqual(clean_string("Example [text]]"), "Example [text]")
        self.assertEqual(clean_string("[Example] text"), "[Example] text")

    def test_unbalanced_braces(self):
        self.assertEqual(clean_string(
            "{This} is a } test"), "{This} is a test")
        self.assertEqual(clean_string("{This {is a test"), "This is a test")

    def test_unbalanced_quotes(self):
        self.assertEqual(clean_string("\"Quote\" extra\""), "\"Quote\" extra")
        self.assertEqual(clean_string("Extra \"quote"), "Extra quote")
        self.assertEqual(clean_string(
            "\"Quote\" extra"), "\"Quote\" extra")
        self.assertEqual(clean_string("\"Extra quote"), "Extra quote")
        self.assertEqual(clean_string("Quote extra\""), "Quote extra")

    def test_nested_unbalanced(self):
        self.assertEqual(clean_string(
            "{[\"Mismatched\")]}"), "{[\"Mismatched\"]}")
        self.assertEqual(clean_string(
            "This \"(is [a {[test])"), "This (is a [test])")

    def test_trailing_apostrophe(self):
        self.assertEqual(clean_string("Hello World'"), "Hello World")
        self.assertEqual(clean_string("Hello World’"), "Hello World")

    def test_leading_and_trailing_commas(self):
        self.assertEqual(clean_string(",Hello World,"), "Hello World")
        self.assertEqual(clean_string(",Hello World"), "Hello World")
        self.assertEqual(clean_string("Hello World,"), "Hello World")

    def test_leading_and_trailing_double_quotes(self):
        self.assertEqual(clean_string("\"Hello World\""), "Hello World")
        self.assertEqual(clean_string("\"Hello World"), "Hello World")
        self.assertEqual(clean_string("Hello World\""), "Hello World")

    def test_leading_and_trailing_single_quotes(self):
        self.assertEqual(clean_string("'Hello World'"), "Hello World")
        self.assertEqual(clean_string("'Hello World"), "Hello World")
        self.assertEqual(clean_string("Hello World'"), "Hello World")

    def test_leading_and_trailing_spaces(self):
        self.assertEqual(clean_string(" Hello World "), "Hello World")
        self.assertEqual(clean_string(" Hello World"), "Hello World")
        self.assertEqual(clean_string("Hello World "), "Hello World")

    def test_preserve_newlines(self):
        self.assertEqual(clean_string(
            "Hello\nWorld"), "Hello\nWorld")
        self.assertEqual(clean_string(
            "Hello\nWorld\n"), "Hello\nWorld")

    def test_no_change_needed(self):
        self.assertEqual(clean_string("This is a test."), "This is a test.")
        self.assertEqual(clean_string(
            "This (is [a {test}])"), "This (is [a {test}])")
        self.assertEqual(clean_string(
            "(Balanced) [text]"), "(Balanced) [text]")

    def test_edge_cases(self):
        self.assertEqual(clean_string(""), "")
        self.assertEqual(clean_string(" , "), ",")
        self.assertEqual(clean_string("\",\""), ",")


class TestCleanSample(unittest.TestCase):

    def test_quotes(self):
        self.assertEqual(clean_sample("\"Quote\" extra\""), "\"Quote\" extra")
        self.assertEqual(clean_sample("Extra \"quote"), "Extra quote")
        self.assertEqual(clean_sample(
            "\"Quote\" extra"), "\"Quote\" extra")
        self.assertEqual(clean_sample("\"Extra quote"), "Extra quote")
        self.assertEqual(clean_sample("Quote extra\""), "Quote extra")
        self.assertEqual(clean_sample("”Enclosed quote”"), "Enclosed quote")

    def test_edge_cases(self):
        self.assertEqual(clean_sample(""), "")
        self.assertEqual(clean_sample(","), "")
        self.assertEqual(clean_sample(" , "), "")
        self.assertEqual(clean_sample("\",\""), ",")


class TestDataProcessingFunctions(unittest.TestCase):
    def test_analyze_sentence_endings(self):
        data = ["Hello.", "How are you?", "Fine!"]
        result = analyze_sentence_endings(data)
        expected = Counter({'.': 1, '?': 1, '!': 1})
        self.assertEqual(result, expected)

    def test_balance_languages(self):
        data = [{'language': 'en'}, {'language': 'tl'}, {'language': 'en'}]
        result = balance_languages(data)
        expected = {'en': 2/3, 'tl': 1/3}
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()

import unittest
from sentence import (
    process_sentence_newlines,
    adaptive_split,
    is_ordered_list_marker,
    is_ordered_list_sentence,
    count_sentences,
    get_sentences
)
from words import count_words


def mock_count_tokens(tokenizer):
    def count_tokens(text):
        return len(tokenizer.encode(text))

    return count_tokens


class TestProcessSentences(unittest.TestCase):

    def test_with_newline_token(self):
        sentences = ['Sentence before1.NEWLINE_TOKEN',
                     'Sentence before2. NEWLINE_TOKEN1. ', 'Sentence after.']

        result = process_sentence_newlines(sentences)
        expected = ['Sentence before1.', 'NEWLINE_TOKEN Sentence before2.',
                    'NEWLINE_TOKEN1. Sentence after.']
        self.assertEqual(result, expected)

    def test_with_multiple_newline_tokens(self):
        sentences = ['Sentence before1. NEWLINE_TOKENNEWLINE_TOKEN ',
                     'Sentence before2.NEWLINE_TOKENNEWLINE_TOKEN1. ',
                     'Sentence after.NEWLINE_TOKENNEWLINE_TOKEN ']

        result = process_sentence_newlines(sentences)
        expected = ['Sentence before1.', 'NEWLINE_TOKENNEWLINE_TOKEN Sentence before2.',
                    'NEWLINE_TOKENNEWLINE_TOKEN1. Sentence after.']
        self.assertEqual(result, expected)

    def test_without_newline_token(self):
        sentences = ['Sentence before1.', 'Sentence before2.',
                     'Sentence after.']

        result = process_sentence_newlines(sentences)
        expected = ['Sentence before1.', 'Sentence before2.',
                    'Sentence after.']
        self.assertEqual(result, expected)

    def test_empty_input(self):
        sentences = []

        result = process_sentence_newlines(sentences)
        expected = []
        self.assertEqual(result, expected)


class TestAdaptiveSplit_English(unittest.TestCase):
    def test_empty_input(self):
        text = ""
        expected_segments = []
        max_segment_tokens = 10
        segments = adaptive_split(
            text, count_words, max_segment_tokens)
        self.assertEqual(segments, expected_segments)

    def test_single_sentence(self):
        text = "The Philippines, officially the Republic of the Philippines, is an archipelagic country in Southeast Asia."
        expected_segments = [text]
        max_segment_tokens = 50
        segments = adaptive_split(
            text, count_words, max_segment_tokens)

        self.assertEqual(segments, expected_segments)

    def test_long_paragraph(self):
        text = "The Philippines, officially the Republic of the Philippines, is an archipelagic country in Southeast Asia. In the western Pacific Ocean, it consists of 7,641 islands which are broadly categorized in three main geographical divisions from north to south: Luzon, Visayas, and Mindanao. The Philippines is bounded by the South China Sea to the west, the Philippine Sea to the east, and the Celebes Sea to the south. It shares maritime borders with Taiwan to the north, Japan to the northeast, Palau to the east and southeast, Indonesia to the south, Malaysia to the southwest, Vietnam to the west, and China to the northwest. It is the world's twelfth-most-populous country, with diverse ethnicities and cultures. Manila is the country's capital, and its most populated city is Quezon City; both are within Metro Manila."

        max_segment_tokens = 50
        segments = adaptive_split(
            text, count_words, max_segment_tokens)

        segment_lengths = [count_words(segment)
                           for segment in segments]

        self.assertTrue(all(segment_length <= max_segment_tokens
                            for segment_length in segment_lengths))
        self.assertGreater(len(segments), 1)

    def test_ordered_list(self):
        text = "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
        expected = [
            '1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats.',
            'This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.',
            '\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health.',
            'Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.',
            '\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being.',
            'It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function.',
            'Aim for 7-9 hours of sleep each night.']

        segments = adaptive_split(text)

        self.assertGreater(len(segments), 1)
        # Assert preserve newlines
        self.assertTrue("\n" in ''.join(segments))

        # Assert the segments are the same as the expected
        self.assertEqual(segments, expected)

    def test_mixed_newlines(self):
        text = "Dear customer,\n\nFirst and foremost, I would like to sincerely apologize for any inconvenience or frustration you have experienced with our product. It is important to us that our customers are fully satisfied with their purchases and we take concerns like yours very seriously.\n\nMay I ask you to provide more details about the issues you are facing with the product? That way, we can work together towards finding the best possible solution to address your concerns.\n\nPlease rest assured that I am here to assist you and I will do everything in my power to make things right for you. We value your business and appreciate the opportunity to make things right.\n\nThank you for bringing this matter to our attention.\n\nSincerely,\nAI Assistant"

        max_segment_tokens = None
        segments = adaptive_split(
            text, count_words, max_segment_tokens)

        all_segments_text = ' '.join(segments)

        # Assertion to count equal amount of newlines
        self.assertEqual(text.count('\n'), all_segments_text.count('\n'))

    def test_low_max_segment_tokens(self):
        text = "The Philippines, officially the Republic of the Philippines, is an archipelagic country in Southeast Asia. In the western Pacific Ocean, it consists of 7,641 islands which are broadly categorized in three main geographical divisions from north to south: Luzon, Visayas, and Mindanao."

        max_segment_tokens = 10
        segments = adaptive_split(
            text, count_words, max_segment_tokens)

        # Assertion to ensure the function returns a non empty list of strings
        self.assertTrue(len(segments) > 0,
                        "The function should return a non empty list of strings")

    def test_mixed_symbols(self):
        text = "The Tertiary stage / economic level: involves providing services to consumers such as baby care, film and banking."
        expected = [
            text
        ]

        segments = adaptive_split(text)

        self.assertEqual(segments, expected)


class TestIsOrderedListMarker(unittest.TestCase):
    def test_list_marker_without_words(self):
        self.assertTrue(is_ordered_list_marker("1."))
        self.assertTrue(is_ordered_list_marker("1.)"))
        self.assertTrue(is_ordered_list_marker("1.) "))

    def test_list_marker_with_words(self):
        self.assertFalse(is_ordered_list_marker("1.This is a list item"))
        self.assertFalse(is_ordered_list_marker("1. This is a list item"))
        self.assertFalse(is_ordered_list_marker("1.) This is a list item"))
        self.assertFalse(is_ordered_list_marker("1.) This is a list item "))


class TestIsOrderedListSentence(unittest.TestCase):

    def test_ordered_numbers(self):
        self.assertTrue(is_ordered_list_sentence("1. This is a list item"))
        self.assertTrue(is_ordered_list_sentence("2. Another item"))
        self.assertTrue(is_ordered_list_sentence("10. Tenth item"))

    def test_ordered_alphabets(self):
        self.assertTrue(is_ordered_list_sentence("a.Alphabetical marker 1"))
        self.assertTrue(is_ordered_list_sentence("B.)Alphabetical marker 2"))
        self.assertTrue(is_ordered_list_sentence("C) Alphabetical marker 3"))
        self.assertTrue(is_ordered_list_sentence("D. Alphabetical marker 4"))

    def test_ordered_roman_numerals(self):
        self.assertTrue(is_ordered_list_sentence("i. Roman numeral marker 1"))
        self.assertTrue(is_ordered_list_sentence("ii. Roman numeral marker 2"))
        self.assertTrue(is_ordered_list_sentence(
            "iii. Roman numeral marker 3"))
        self.assertTrue(is_ordered_list_sentence("iv. Roman numeral marker 4"))

    def test_without_ordered_list_marker(self):
        self.assertFalse(is_ordered_list_sentence("This is not a list item"))
        self.assertFalse(is_ordered_list_sentence(
            "1- This is not a list item"))
        self.assertFalse(is_ordered_list_sentence("No number at start"))

    def test_empty_string(self):
        self.assertFalse(is_ordered_list_sentence(""))

    def test_non_numeric_marker(self):
        self.assertFalse(is_ordered_list_sentence(
            "First. This is not a numeric list item"))
        self.assertFalse(is_ordered_list_sentence("Second item"))

    def test_newline_marker(self):
        self.assertTrue(is_ordered_list_sentence(
            "\n1. This is a list item"))
        self.assertTrue(is_ordered_list_sentence(
            "1. This is a list item\n\n"))
        self.assertTrue(is_ordered_list_sentence(
            "\n1. This is a list item\n1. This is another list item"))

    def test_list_marker_without_words(self):
        self.assertFalse(is_ordered_list_sentence("1."))
        self.assertFalse(is_ordered_list_sentence("1.)"))
        self.assertFalse(is_ordered_list_sentence("1.) "))


class TestSentenceCounting(unittest.TestCase):

    def test_count_sentences_with_list_markers(self):
        text = "Here are three tips for staying healthy\n1. Eat a balanced diet 2. Exercise regularly 3. Get enough sleep"
        result = count_sentences(text)
        self.assertEqual(result, 4)

    def test_count_sentences_with_roman_numerals(self):
        text = "Important points:\ni. Be kind.\nii. Work hard.\niii. Stay humble."
        result = count_sentences(text)
        self.assertEqual(result, 4)

    def test_count_sentences_regular_text(self):
        text = "This is a sentence. This is another sentence."
        result = count_sentences(text)
        self.assertEqual(result, 2)

    def test_count_sentences_with_mixed_content(self):
        text = "First point: a. Always be learning. Regular sentence here."
        result = count_sentences(text)
        self.assertEqual(result, 3)


class TestGetSentences(unittest.TestCase):

    def test_get_n_sentences(self):
        text = "This is a sentence. This is another sentence. This is the third sentence."
        result = get_sentences(text, 2)
        expected = ["This is a sentence.", "This is another sentence."]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()

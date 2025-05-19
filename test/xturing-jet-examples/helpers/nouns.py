import unittest
import spacy
import re


class WordFinder:
    def __init__(self):
        # Load the spaCy model for English. You might need to download it first using spacy.cli.download('en_core_web_sm')
        self.nlp = spacy.load("en_core_web_sm")

    def analyze_text(self, text):
        doc = self.nlp(text)
        results = {
            "entities": [],
            "noun_chunks": [],
            "sentences": [],
            "tokens": [],
            "token_children": [],
        }

        # Extract entities
        results["entities"] = [entity.text for entity in doc.ents]

        # Extract noun chunks
        results["noun_chunks"] = [chunk.text for chunk in doc.noun_chunks]

        # Extract sentences
        results["sentences"] = [sent.text for sent in doc.sents]

        # Extract tokens and their children
        for token in doc:
            results["token_children"].append(
                {token.text: [child.text for child in token.children]})

        return results

    def find_words_and_word_groups(self, text):
        """
        Find unique phrases in the given text.
        """
        # results = self.analyze_text(text)

        # Process the text using the NLP model
        doc = self.nlp(text)

        # Extract entities and noun chunks; both are often significant phrases
        entities = [entity.text for entity in doc.ents]

        # noun_chunks should include nouns combined by PUNCT and CONJ

        noun_chunks = [chunk.text for chunk in doc.noun_chunks]

        # Combine and remove duplicates while preserving order
        combined = entities + noun_chunks

        # Remove single words that are not proper nouns
        combined = [phrase for phrase in combined if len(
            phrase.split(" ")) > 1 or self.nlp(phrase)[0].pos_ == "PROPN"]

        # Remove first words from multiple word strings that don't match any of the pos tags below
        pos_tags = ["NOUN", "PROPN", "ADJ", "NUM"]
        combined = [phrase.split(" ", 1)[1] if self.nlp(
            phrase.split(" ", 1)[0])[0].pos_ not in pos_tags else phrase for phrase in combined]

        # Remove phrases that start with an adverb
        combined = [phrase for phrase in combined if not self.nlp(
            phrase.split(" ", 1)[0])[0].pos_ == "ADV"]

        unique_phrases = list(dict.fromkeys(combined))

        # Sort phrases by index in text
        unique_phrases.sort(key=lambda phrase: text.find(phrase))

        # Remove phrases that doesn't have the first word matching any of the pos tags
        unique_phrases = [phrase for phrase in unique_phrases if self.nlp(
            phrase.split(" ", 1)[0])[0].pos_ in pos_tags]

        return unique_phrases


class TestWordFinder(unittest.TestCase):
    def setUp(self):
        self.word_finder = WordFinder()

    def test_sentence(self):
        text = "The Philippines, officially the Republic of the Philippines, is an archipelagic country in Southeast Asia."
        expected = ["Philippines", "Republic of the Philippines",
                    "archipelagic country", "Southeast Asia"]
        result = self.word_finder.find_words_and_word_groups(text)
        self.assertEqual(result, expected)

    def test_paragraph(self):
        text = "The Philippines, officially the Republic of the Philippines, is an archipelagic country in Southeast Asia."
        expected = ['Philippines', 'Republic of the Philippines',
                    'archipelagic country', 'Southeast Asia']
        result = self.word_finder.find_words_and_word_groups(text)
        self.assertEqual(result, expected)

    def test_paragraph_2(self):
        text = "Manila is the country's capital, and its most populated city is Quezon City; both are within Metro Manila."
        expected = ['Manila', "country's capital",
                    'Quezon City', 'Metro Manila']
        result = self.word_finder.find_words_and_word_groups(text)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

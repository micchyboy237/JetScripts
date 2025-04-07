import unittest
from transformers import pipeline


class TestNLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nli = pipeline("text-classification",
                           model="facebook/bart-large-mnli")

    def test_entailment(self):
        sample = "A man is writing code on his laptop."
        hypothesis = "Someone is using a computer."
        expected = "ENTAILMENT"
        result = self.nli(f"{sample} </s> {hypothesis}")[0]['label'].upper()
        self.assertEqual(result, expected)

    def test_contradiction(self):
        sample = "The kitchen lights are turned off."
        hypothesis = "The room is brightly lit."
        expected = "CONTRADICTION"
        result = self.nli(f"{sample} </s> {hypothesis}")[0]['label'].upper()
        self.assertEqual(result, expected)

    def test_neutral(self):
        sample = "A teacher is explaining a math problem."
        hypothesis = "The students understand everything."
        expected = "NEUTRAL"
        result = self.nli(f"{sample} </s> {hypothesis}")[0]['label'].upper()
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

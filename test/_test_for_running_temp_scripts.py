import unittest

from for_running_temp_scripts import get_bm25_similarities


class TestBM25Similarity(unittest.TestCase):
    def test_similarity(self):
        queries = ["running fast", "goal oriented"]
        documents = ["I am running at a fast pace", "I have a goal in mind"]
        ids = ["1", "2"]

        result = get_bm25_similarities(queries, documents, ids)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()

import unittest
from dataset import distribute_evenly


class TestTranslationFunctions(unittest.TestCase):
    def test_distribute_evenly(self):
        # Test to ensure that items with the same label are distributed evenly
        data = [{"label": "A"}, {"label": "A"}, {"label": "B"},
                {"label": "B"}, {"label": "C"}, {"label": "C"}]
        distributed_data = distribute_evenly(data)

        # Check that no consecutive items have the same label
        for i in range(len(distributed_data) - 1):
            self.assertNotEqual(
                distributed_data[i]['label'], distributed_data[i + 1]['label'])

        # Test with all identical labels
        data_identical = [{"label": "A"}, {"label": "A"}, {"label": "A"}]
        distributed_identical = distribute_evenly(data_identical)
        for i in range(len(distributed_identical) - 1):
            self.assertEqual(
                distributed_identical[i]['label'], distributed_identical[i + 1]['label'])

        # Test with unique labels
        data_unique = [{"label": "A"}, {"label": "B"}, {"label": "C"}]
        distributed_unique = distribute_evenly(data_unique)
        for i in range(len(distributed_unique) - 1):
            self.assertNotEqual(
                distributed_unique[i]['label'], distributed_unique[i + 1]['label'])


if __name__ == '__main__':
    unittest.main()

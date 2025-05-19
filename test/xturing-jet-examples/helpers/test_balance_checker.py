import unittest
from balance_checker import get_distributions


class TestProcessData(unittest.TestCase):

    def test_no_filter_no_category(self):
        # Test with no filters and no categories
        mock_data = [
            {'category': 'question and answering', 'content': 'QA content 1'},
            {'category': 'list, name and sequence', 'content': 'List content 1'},
            {'category': 'miscellaneous', 'content': 'Misc content'}
        ]
        result = get_distributions(mock_data)
        self.assertEqual(len(result), 3)

    def test_with_category_filter(self):
        # Test with specific category filter
        mock_data = [
            {'category': 'question and answering', 'content': 'QA content'},
            {'category': 'list, name and sequence', 'content': 'List content'}
        ]
        categories = ['question and answering']
        result = get_distributions(mock_data, categories=categories)
        self.assertEqual(len(result), 1)
        self.assertTrue(
            all(item['category'] == 'question and answering' for item in result))

    def test_with_contains_filter(self):
        # Test with 'contains' string filter
        mock_data = [
            {'category': 'miscellaneous', 'content': 'Content with xyz'},
            {'category': 'miscellaneous', 'content': 'Another content'}
        ]
        filters = ['content:contains:xyz']
        result = get_distributions(mock_data, filters=filters)
        self.assertEqual(len(result), 1)
        self.assertIn('xyz', result[0]['content'])

    def test_with_starts_filter(self):
        # Test with 'starts' string filter
        mock_data = [
            {'category': 'type1', 'content': 'Starts with specific text'},
            {'category': 'type2', 'content': 'Does not start with'}
        ]
        filters = ['content:starts:Starts with']
        result = get_distributions(mock_data, filters=filters)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]['content'].startswith('Starts with'))

    def test_with_ends_filter(self):
        # Test with 'ends' string filter
        mock_data = [
            {'category': 'type1', 'content': 'Ends with specific text'},
            {'category': 'type2', 'content': 'Does not end with'}
        ]
        filters = ['content:ends:with specific text']
        result = get_distributions(mock_data, filters=filters)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]['content'].endswith('with specific text'))

    def test_with_multiple_filters_or_condition(self):
        # Test with multiple filters - OR condition
        mock_data = [
            {'category': 'miscellaneous', 'content': 'Content with xyz'},
            {'category': 'miscellaneous',
                'content': 'Another content starts with Content'}
        ]
        filters = ['content:contains:xyz', 'content:starts:Another']
        result = get_distributions(mock_data, filters=filters)
        # Both items match at least one filter
        self.assertEqual(result, mock_data)

    def test_combination_of_filters_and_categories(self):
        # Test with a combination of filters and categories - OR condition
        mock_data = [
            {'category': 'type1', 'content': 'Content for type1'},
            {'category': 'type2', 'content': 'Another content for type2'},
            {'category': 'type1', 'content': 'Additional content for type1'}
        ]
        categories = ['type1']
        filters = ['content:contains:content for']
        result = get_distributions(
            mock_data, categories=categories, filters=filters)
        expected = [
            {'category': 'type1', 'content': 'Content for type1'},
            {'category': 'type1', 'content': 'Additional content for type1'}
        ]

        # Assert items match all type1 category and content filter
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()

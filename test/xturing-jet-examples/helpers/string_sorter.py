import unittest
from tqdm import tqdm


class StringSorter:
    @staticmethod
    def get_first_word(string):
        return string.split()[0] if string else ''

    @staticmethod
    def sort_by_first_word(strings):
        return sorted(strings, key=StringSorter.get_first_word)

    @staticmethod
    def sort_and_spread_by_first_word(strings):
        sorted_strings = StringSorter.sort_by_first_word(strings)
        spread_strings = [sorted_strings.pop(0)]

        pbar = tqdm(total=len(sorted_strings), desc="Sorting strings...")
        while sorted_strings:
            pbar.update(1)
            for i, string in enumerate(sorted_strings):
                if StringSorter.get_first_word(string) != StringSorter.get_first_word(spread_strings[-1]):
                    spread_strings.append(sorted_strings.pop(i))
                    break
            else:
                spread_strings.append(sorted_strings.pop(0))

        return spread_strings

    @staticmethod
    def has_common_substring(str1, str2, min_length=3):
        if str1 == str2:
            return True

        for i in range(len(str1)):
            for j in range(min_length, len(str1) - i + 1):
                if str1[i:i+j] in str2:
                    return True
        return False

    @staticmethod
    def get_combined_string_from_keys(obj, keys):
        return ' '.join(str(obj[key]) for key in keys if key in obj)

    @staticmethod
    def sort_and_spread_by_substring(items, sort_keys=None):
        # Sort by sort_keys first, then by first word or string itself
        if sort_keys:
            items = sorted(items, key=lambda x: (StringSorter.get_combined_string_from_keys(
                x, sort_keys), StringSorter.get_combined_string_from_keys(x, [sort_keys[0]])))
        else:
            items = StringSorter.sort_by_first_word(items)

        spread_items = [items.pop(0)]

        for i in range(len(items)):
            for j, item in enumerate(items):
                item_str = StringSorter.get_combined_string_from_keys(
                    item, sort_keys) if sort_keys else item
                last_item_str = StringSorter.get_combined_string_from_keys(
                    spread_items[-1], sort_keys) if sort_keys else spread_items[-1]

                if not StringSorter.has_common_substring(last_item_str, item_str):
                    spread_items.append(items.pop(j))
                    break
            else:
                spread_items.append(items.pop(0))

        return spread_items


# Unit tests
class TestStringSorter(unittest.TestCase):
    def test_sort_by_first_word(self):
        sorter = StringSorter()
        strings = ["Red apple", "Blue sky",
                   "Apple pie", "Red rose", "Blue ocean"]
        sorted_strings = sorter.sort_by_first_word(strings)
        self.assertEqual(sorted_strings, [
                         "Apple pie", "Blue ocean", "Blue sky", "Red apple", "Red rose"])

    def test_sort_and_spread_by_first_word(self):
        sorter = StringSorter()
        strings = ["Red apple", "Red rose", "Blue ocean",
                   "Blue sky", "Apple pie"]
        spread_strings = sorter.sort_and_spread_by_first_word(strings)
        for i in range(len(spread_strings) - 1):
            self.assertNotEqual(spread_strings[i].split()[
                                0], spread_strings[i + 1].split()[0])

    def test_sort_and_spread_by_substring_with_strings(self):
        sorter = StringSorter()
        strings = ["apple pie", "pineapple",
                   "blueberry", "apple juice", "berry tart"]
        spread_strings = sorter.sort_and_spread_by_substring(strings)
        for i in range(len(spread_strings) - 1):
            self.assertFalse(sorter.has_common_substring(
                spread_strings[i], spread_strings[i + 1]))

    def test_sort_and_spread_by_substring_with_objects(self):
        sorter = StringSorter()
        objects = [
            {"name": "apple pie", "category": "dessert"},
            {"name": "pineapple", "category": "fruit"},
            {"name": "blueberry", "category": "fruit"},
            {"name": "apple juice", "category": "beverage"},
            {"name": "berry tart", "category": "dessert"}
        ]
        sort_keys = ["name"]
        spread_objects = sorter.sort_and_spread_by_substring(
            objects, sort_keys)
        for i in range(len(spread_objects) - 1):
            item_str1 = sorter.get_combined_string_from_keys(
                spread_objects[i], sort_keys)
            item_str2 = sorter.get_combined_string_from_keys(
                spread_objects[i + 1], sort_keys)
            self.assertFalse(sorter.has_common_substring(item_str1, item_str2))


if __name__ == "__main__":
    unittest.main()

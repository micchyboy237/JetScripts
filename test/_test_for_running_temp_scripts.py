import unittest
from for_running_temp_scripts import construct_google_query


class TestConstructGoogleQuery(unittest.TestCase):

    def test_basic_query(self):
        result = construct_google_query(
            "I'll Become a Villainess Who Goes Down in History")
        expected = 'https://www.google.com/search?q=%22I%27ll+Become+a+Villainess+Who+Goes+Down+in+History%22+anime'
        self.assertEqual(result, expected)

    def test_query_with_properties(self):
        result = construct_google_query("I'll Become a Villainess Who Goes Down in History",
                                        properties=["seasons", "episodes", "synopsis", "genre"])
        expected = 'https://www.google.com/search?q=%22I%27ll+Become+a+Villainess+Who+Goes+Down+in+History%22+anime+seasons+episodes+synopsis+genre'
        self.assertEqual(result, expected)

    def test_query_with_site(self):
        result = construct_google_query(
            "I'll Become a Villainess Who Goes Down in History", site="myanimelist.net")
        expected = 'https://www.google.com/search?q=%22I%27ll+Become+a+Villainess+Who+Goes+Down+in+History%22+anime+site:myanimelist.net'
        self.assertEqual(result, expected)

    def test_query_with_exclusions(self):
        result = construct_google_query(
            "I'll Become a Villainess Who Goes Down in History", exclude=["reddit", "forum"])
        expected = 'https://www.google.com/search?q=%22I%27ll+Become+a+Villainess+Who+Goes+Down+in+History%22+anime+-reddit+-forum'
        self.assertEqual(result, expected)

    def test_full_query(self):
        result = construct_google_query("I'll Become a Villainess Who Goes Down in History",
                                        properties=[
                                            "release date", "end date"],
                                        site="anilist.co",
                                        exclude=["reddit", "forum"])
        expected = 'https://www.google.com/search?q=%22I%27ll+Become+a+Villainess+Who+Goes+Down+in+History%22+anime+release+date+end+date+site:anilist.co+-reddit+-forum'
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

from rake_nltk import Rake

# Initialize RAKE
rake = Rake(max_length=3)  # Set max phrase length for anime titles

# Sample text
text = "I enjoyed Cowboy Bebop and Neon Genesis Evangelion."

# Extract keywords
rake.extract_keywords_from_text(text)
keywords = rake.get_ranked_phrases()
anime_titles = [kw for kw in keywords if kw in [
    "Cowboy Bebop", "Neon Genesis Evangelion"]]
print(anime_titles)  # Output: ['Cowboy Bebop', 'Neon Genesis Evangelion']

# Pytest test


def test_rake_extraction():
    text = "I enjoyed Cowboy Bebop."
    rake.extract_keywords_from_text(text)
    assert "Cowboy Bebop" in rake.get_ranked_phrases()

# Sample usage
from jet.wordnet.keywords.keyword_extraction import extract_keywords


# Sample usage
if __name__ == "__main__":
    text = "I love watching Naruto and Attack on Titan. Have you seen One Piece?"
    query = "anime"
    anime_titles = extract_keywords(
        text,
        query=query,
        # stop_words=['love', 'watching', 'seen']
    )
    # Expected: ['naruto', 'attack on titan', 'one piece', ...]
    print(anime_titles)

# Pytest test

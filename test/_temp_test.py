from keybert import KeyBERT

from jet.logger import logger
from jet.transformers.formatters import format_json

# Initialize KeyBERT
model = KeyBERT("all-mpnet-base-v2")  # Lightweight for Mac M1

# Sample text
text = "I love watching Naruto and Attack on Titan. Have you seen One Piece?"

# Extract keywords guided by query
query = "anime"
keywords = model.extract_keywords(
    text,
    keyphrase_ngram_range=(1, 3),
    top_n=5,
    use_mmr=True,
    diversity=0.5
    # Removed candidates=["anime"] to avoid interfering with keyphrase extraction
)
anime_titles = [kw[0] for kw in keywords]
# Expected output: ['naruto', 'attack on titan', 'one piece', ...]
logger.success(format_json(anime_titles))

# Pytest test


def test_keybert_query_extraction():
    text = "I love watching Naruto and Attack on Titan."
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        top_n=2,
        use_mmr=True,
        diversity=0.5
        # Removed candidates=["anime"]
    )
    extracted = [kw[0] for kw in keywords]
    assert set(['attack on titan', 'naruto']).issubset(extracted)


if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])

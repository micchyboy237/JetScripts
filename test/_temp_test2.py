import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    """Preprocess text: lowercase, remove punctuation, and stopwords."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [
        t for t in tokens if t not in string.punctuation and t not in stop_words]
    return ' '.join(tokens)


def text_diversity_score(text1, text2, semantic_weight=0.5):
    """
    Compute diversity score between two texts (0 = identical, 1 = completely diverse).

    Args:
        text1 (str): First text.
        text2 (str): Second text.
        semantic_weight (float): Weight for semantic similarity (0 to 1).

    Returns:
        float: Diversity score between 0 and 1.
    """
    if not text1.strip() or not text2.strip():
        return 1.0

    # Lexical diversity using TF-IDF
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
    lexical_similarity = cosine_similarity(
        tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    lexical_diversity = 1 - lexical_similarity

    # Semantic diversity using Sentence-BERT
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text1, text2])
    semantic_similarity = cosine_similarity(
        [embeddings[0]], [embeddings[1]])[0][0]
    semantic_diversity = 1 - semantic_similarity

    # Combine scores
    final_score = (semantic_weight * semantic_diversity) + \
        ((1 - semantic_weight) * lexical_diversity)
    return np.clip(final_score, 0, 1)


def sort_texts_for_diversity(texts, semantic_weight=0.5):
    """
    Sort a list of texts to maximize diversity between adjacent texts.

    Args:
        texts (list): List of text strings.
        semantic_weight (float): Weight for semantic diversity in scoring.

    Returns:
        list: Sorted list of texts.
    """
    if not texts:
        return []
    if len(texts) == 1:
        return texts

    # Initialize result with the first text
    result = [texts[0]]
    remaining = texts[1:].copy()

    while remaining:
        last_text = result[-1]
        # Compute diversity scores between the last selected text and all remaining texts
        diversity_scores = [
            text_diversity_score(last_text, text, semantic_weight) for text in remaining
        ]
        # Select the text with the highest diversity score
        max_diversity_idx = np.argmax(diversity_scores)
        result.append(remaining.pop(max_diversity_idx))

    return result

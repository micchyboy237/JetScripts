import logging
from typing import List, Tuple
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np
from typing import TypedDict

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Document(TypedDict):
    text: str
    label: str


def scrape_web_page(url: str) -> str:
    """Scrape the main content from a web page."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logger.error(
                f"Failed to retrieve {url}. Status code: {response.status_code}")
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove common noise elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        text = soup.get_text(separator=' ', strip=True)
        logger.info(f"Successfully scraped content from {url}")
        return text
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return ""


def preprocess_text(text: str) -> str:
    """Preprocess text by tokenizing, removing stopwords, and stemming."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum()
              and token not in stop_words]

    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed)


def prepare_dataset() -> List[Document]:
    """Create a synthetic dataset for demonstration."""
    # In a real scenario, this would be loaded from a file or database
    dataset = [
        {"text": "This is a legitimate article about machine learning", "label": "normal"},
        {"text": "Buy cheap products now! Click here for discounts", "label": "noise"},
        {"text": "Deep learning models for text classification", "label": "normal"},
        {"text": "Free offers! Visit our site for deals", "label": "noise"},
    ]
    return [
        {"text": preprocess_text(doc["text"]), "label": doc["label"]}
        for doc in dataset
    ]


def train_classifier(documents: List[Document]) -> Pipeline:
    """Train an SVM classifier for noise detection."""
    texts = [doc["text"] for doc in documents]
    labels = [doc["label"] for doc in documents]

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('svm', SVC(kernel='linear', probability=True))
    ])

    pipeline.fit(texts, labels)
    logger.info("Classifier trained successfully")
    return pipeline


def classify_document(classifier: Pipeline, text: str) -> str:
    """Classify a document as noise or normal."""
    processed_text = preprocess_text(text)
    prediction = classifier.predict([processed_text])[0]
    return prediction


# Example usage
if __name__ == "__main__":
    # Prepare dataset
    dataset = prepare_dataset()

    # Train classifier
    classifier = train_classifier(dataset)

    # Example: Scrape and classify a web page
    test_url = "http://quotes.toscrape.com"  # Example URL
    scraped_text = scrape_web_page(test_url)
    if scraped_text:
        result = classify_document(classifier, scraped_text)
        logger.info(f"Classification result for {test_url}: {result}")

import os
import logging
import numpy as np
from typing import List, Tuple, TypedDict
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

from jet.models.utils import resolve_model_value
from jet.models.model_types import EmbedModelType


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextClusterer:
    def __init__(self, model_name: str = "all-MiniLM-L12-v2", n_clusters: int = 2):
        self.model = SentenceTransformer(
            model_name, device="cpu", backend="onnx")
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        logger.info(
            f"Initialized TextClusterer with model: {model_name}, n_clusters: {n_clusters}")

    def encode_texts(self, texts: List[str]) -> NDArray[np.float64]:
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(
                f"Encoded {len(texts)} texts into embeddings of shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def fit_predict(self, texts: List[str]) -> NDArray[np.int_]:
        try:
            embeddings = self.encode_texts(texts)
            clusters = self.clusterer.fit_predict(embeddings)
            logger.info("Clustering completed successfully")
            return clusters
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            raise


if __name__ == "__main__":
    texts = [
        "NumPy and pandas are great Python libraries.",
        "Java is great for enterprise applications.",
        "Scikit-learn is a machine learning library.",
        "Matplotlib is a Python plotting library."
    ]
    clusterer = TextClusterer()
    clusters = clusterer.fit_predict(texts)
    for text, cluster in zip(texts, clusters):
        print(f"Text: {text[:30]}..., Cluster: {cluster}")


class AnomalyDetector:
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        """Initialize anomaly detector with Sentence Transformer and Isolation Forest."""
        device = "cpu"
        self.model = SentenceTransformer(
            model_name, device=device, backend="onnx")
        self.detector = IsolationForest(contamination=0.4, random_state=42)
        logger.info(
            f"Initialized AnomalyDetector with model: {model_name}, device: {device}")

    def encode_texts(self, texts: List[str]) -> NDArray[np.float64]:
        """Encode texts into embeddings."""
        try:
            if not texts or not all(text.strip() for text in texts):
                raise ValueError("Empty or invalid texts provided")
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(
                f"Encoded {len(texts)} texts into embeddings of shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def fit_predict(self, texts: List[str]) -> NDArray[np.int_]:
        """Detect anomalies in texts (-1 for anomaly, 1 for normal)."""
        try:
            embeddings = self.encode_texts(texts)
            anomalies = self.detector.fit_predict(embeddings)
            logger.info("Anomaly detection completed successfully")
            return anomalies
        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            raise


if __name__ == "__main__":
    # Real-world example: Detect spam comments on a social media platform
    comments = [
        "Great post, thanks for sharing!",
        "Really helpful information.",
        "Buy cheap shoes now! Click here!",
        "Loved the content, keep it up!",
        "Free iPhone giveaway, claim now!"
    ]
    detector = AnomalyDetector()
    anomaly_labels = detector.fit_predict(comments)
    logger.info("Anomaly detection results for comments:")
    for comment, label in zip(comments, anomaly_labels):
        status = "Anomaly (Spam)" if label == -1 else "Normal"
        logger.info(f"Comment: {comment[:50]}..., Status: {status}")


class QueryRanker:
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        """Initialize query ranker with Sentence Transformer."""
        device = "cpu"
        self.model = SentenceTransformer(
            model_name, device=device, backend="onnx")
        logger.info(
            f"Initialized QueryRanker with model: {model_name}, device: {device}")

    def encode_texts(self, texts: List[str]) -> NDArray[np.float64]:
        """Encode texts into embeddings."""
        try:
            if not texts or not all(text.strip() for text in texts):
                raise ValueError("Empty or invalid texts provided")
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(
                f"Encoded {len(texts)} texts into embeddings of shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def rank_contexts(self, query: str, contexts: List[str]) -> List[Tuple[str, float]]:
        """Rank contexts by relevance to the query."""
        try:
            if not query.strip() or not contexts:
                raise ValueError("Empty query or contexts provided")
            query_emb = self.encode_texts([query])[0]
            context_embs = self.encode_texts(contexts)
            scores = util.cos_sim(query_emb, context_embs).flatten()
            ranked = sorted(zip(contexts, scores),
                            key=lambda x: x[1], reverse=True)
            logger.info(
                f"Ranked {len(contexts)} contexts for query: {query[:50]}...")
            return [(context, float(score)) for context, score in ranked]
        except Exception as e:
            logger.error(f"Error ranking contexts: {e}")
            raise


if __name__ == "__main__":
    # Real-world example: Rank customer support articles for a query
    query = "How to reset my password"
    articles = [
        "To reset your password, go to the login page and click 'Forgot Password'.",
        "Our product features include secure encryption and fast performance.",
        "Follow these steps to change your account email address.",
        "Password reset instructions: check your email for a link."
    ]
    ranker = QueryRanker()
    ranked_articles = ranker.rank_contexts(query, articles)
    logger.info("Ranking results for customer support query:")
    for article, score in ranked_articles:
        logger.info(f"Article: {article[:50]}..., Score: {score:.2f}")


class QueryContextPair(TypedDict):
    query: str
    context: str
    label: int  # 1: Relevant, 0: Irrelevant


class KNNRelevanceClassifier:
    def __init__(self, model_name: EmbedModelType = "all-MiniLM-L6-v2", n_neighbors: int = 3):
        """Initialize k-NN classifier with Sentence Transformer model."""
        device = "cpu"
        model_id = resolve_model_value(model_name)
        self.model = SentenceTransformer(
            model_id, device=device, backend="onnx")
        self.classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors, metric='cosine')
        self.embeddings: NDArray[np.float64] | None = None
        self.labels: NDArray[np.int_] | None = None
        logger.info(
            f"Initialized KNNRelevanceClassifier with model: {model_name}, n_neighbors: {n_neighbors}, device: {device}")

    def encode_pairs(self, pairs: List[QueryContextPair]) -> NDArray[np.float64]:
        """Encode query-context pairs into embeddings."""
        try:
            texts = [
                f"{pair['query']} [SEP] {pair['context']}" for pair in pairs]
            if not all(text.strip() for text in texts):
                raise ValueError("Empty query or context detected")
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(
                f"Encoded {len(pairs)} pairs into embeddings of shape: {embeddings.shape}")
            logger.debug(
                f"Sample embedding (first 5 dims): {embeddings[0][:5]}")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding pairs: {e}")
            raise

    def fit(self, pairs: List[QueryContextPair]) -> None:
        """Store embeddings and labels for k-NN (no training)."""
        try:
            self.embeddings = self.encode_pairs(pairs)
            self.labels = np.array([pair['label'] for pair in pairs])
            logger.debug(f"Label distribution: {np.bincount(self.labels)}")
            self.classifier.fit(self.embeddings, self.labels)
            logger.info("k-NN classifier fitted with stored embeddings")
        except Exception as e:
            logger.error(f"Error fitting k-NN classifier: {e}")
            raise

    def predict(self, query: str, context: str) -> Tuple[int, float]:
        """Predict relevance for a single query-context pair."""
        try:
            if not query.strip() or not context.strip():
                raise ValueError("Query or context cannot be empty")
            # Dummy label
            pair = [{'query': query, 'context': context, 'label': 0}]
            embedding = self.encode_pairs(pair)[0]
            prediction = self.classifier.predict([embedding])[0]
            probability = self.classifier.predict_proba([embedding])[
                0][prediction]
            distances, indices = self.classifier.kneighbors([embedding])
            logger.debug(
                f"Neighbor distances: {distances[0]}, indices: {indices[0]}")
            logger.debug(f"Neighbor labels: {self.labels[indices[0]]}")
            logger.debug(
                f"Predicted relevance: {prediction}, probability: {probability}")
            return prediction, probability
        except Exception as e:
            logger.error(f"Error predicting relevance: {e}")
            raise


# Example usage
if __name__ == "__main__":
    pairs = [
        {'query': 'best Python libraries',
            'context': 'Top Python libraries for data science include NumPy, pandas.', 'label': 1},
        {'query': 'best Python libraries',
            'context': 'Java is great for enterprise applications.', 'label': 0},
        {'query': 'machine learning',
            'context': 'Scikit-learn is a machine learning library.', 'label': 1},
    ]
    classifier = KNNRelevanceClassifier()
    classifier.fit(pairs)
    pred, prob = classifier.predict(
        query='best Python libraries',
        context='Matplotlib is a Python plotting library.'
    )
    print(
        f"Prediction: {'Relevant' if pred == 1 else 'Irrelevant'}, Probability: {prob:.2f}")


class RelevanceRegressor:
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        self.model = SentenceTransformer(
            model_name, device="cpu", backend="onnx")
        self.regressor = LinearRegression()
        logger.info(f"Initialized RelevanceRegressor with model: {model_name}")

    def encode_pairs(self, pairs: List[QueryContextPair]) -> NDArray[np.float64]:
        try:
            texts = [
                f"{pair['query']} [SEP] {pair['context']}" for pair in pairs]
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(
                f"Encoded {len(pairs)} pairs into embeddings of shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding pairs: {e}")
            raise

    def fit(self, pairs: List[QueryContextPair]) -> None:
        try:
            embeddings = self.encode_pairs(pairs)
            scores = np.array([pair['score'] for pair in pairs])
            self.regressor.fit(embeddings, scores)
            logger.info("Regressor fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting regressor: {e}")
            raise

    def predict(self, query: str, context: str) -> float:
        try:
            pair = [{'query': query, 'context': context, 'score': 0.0}]
            embedding = self.encode_pairs(pair)[0]
            score = self.regressor.predict([embedding])[0]
            logger.debug(f"Predicted relevance score: {score}")
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.error(f"Error predicting score: {e}")
            raise


if __name__ == "__main__":
    pairs = [
        {'query': 'best Python libraries',
            'context': 'NumPy and pandas are great.', 'score': 0.9},
        {'query': 'best Python libraries',
            'context': 'Java is great for enterprise apps.', 'score': 0.2},
        {'query': 'machine learning',
            'context': 'Scikit-learn is a machine learning library.', 'score': 0.95},
    ]
    regressor = RelevanceRegressor()
    regressor.fit(pairs)
    score = regressor.predict(query='best Python libraries',
                              context='Matplotlib is a Python plotting library.')
    print(f"Predicted Relevance Score: {score:.2f}")

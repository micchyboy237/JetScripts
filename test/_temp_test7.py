import logging
import os
from typing import List, Tuple, TypedDict
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from numpy.typing import NDArray

from jet.models.model_types import EmbedModelType
from jet.models.utils import resolve_model_value

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

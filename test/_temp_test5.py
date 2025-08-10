# pattern_recognition.py
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prepare_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Load and prepare the Iris dataset."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    target = iris.target
    logger.info("Loaded Iris dataset with %d samples", len(df))
    return df, target


def preprocess_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess data: scale features and split into train/test sets."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    y = load_iris().target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info("Data split: %d train, %d test samples",
                len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def train_knn_model(
    X_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 3
) -> KNeighborsClassifier:
    """Train a k-NN model."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    logger.info("Trained k-NN model with %d neighbors", n_neighbors)
    return model


def evaluate_model(
    model: KNeighborsClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Evaluate the model and return accuracy."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("Model accuracy: %.2f%%", accuracy * 100)
    return accuracy


def main():
    """Main function to run pattern recognition pipeline."""
    df, _ = load_and_prepare_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_knn_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Final model accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()

# test_pattern_recognition.py
import numpy as np
import pytest
from pattern_recognition import (
    load_and_prepare_data,
    preprocess_data,
    train_knn_model,
    evaluate_model,
)
from sklearn.neighbors import KNeighborsClassifier


@pytest.fixture
def sample_data():
    df, target = load_and_prepare_data()
    return df, target


class TestDataPreparation:
    def test_load_and_prepare_data(self, sample_data):
        expected_columns = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
        expected_shape = (150, 4)
        df, target = sample_data
        result_columns = df.columns.tolist()
        result_shape = df.shape
        assert result_columns == expected_columns, f"Expected {expected_columns}, got {result_columns}"
        assert result_shape == expected_shape, f"Expected shape {expected_shape}, got {result_shape}"
        assert len(
            target) == 150, f"Expected 150 target values, got {len(target)}"


class TestPreprocessing:
    def test_preprocess_data(self, sample_data):
        df, _ = sample_data
        X_train, X_test, y_train, y_test = preprocess_data(
            df, test_size=0.2, random_state=42)
        expected_train_size = 120
        expected_test_size = 30
        assert len(
            X_train) == expected_train_size, f"Expected {expected_train_size} train samples, got {len(X_train)}"
        assert len(
            X_test) == expected_test_size, f"Expected {expected_test_size} test samples, got {len(X_test)}"
        assert X_train.shape[1] == 4, f"Expected 4 features, got {X_train.shape[1]}"


class TestModelTraining:
    def test_train_knn_model(self, sample_data):
        df, target = sample_data
        X_train, _, y_train, _ = preprocess_data(df)
        model = train_knn_model(X_train, y_train, n_neighbors=3)
        expected_model_type = KNeighborsClassifier
        result_model_type = type(model)
        assert result_model_type == expected_model_type, f"Expected {expected_model_type}, got {result_model_type}"
        assert model.n_neighbors == 3, f"Expected 3 neighbors, got {model.n_neighbors}"


class TestModelEvaluation:
    def test_evaluate_model(self, sample_data):
        df, target = sample_data
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = train_knn_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        # Reasonable range for Iris dataset
        expected_accuracy_range = (0.8, 1.0)
        assert (
            expected_accuracy_range[0] <= accuracy <= expected_accuracy_range[1]
        ), f"Accuracy {accuracy} outside expected range {expected_accuracy_range}"

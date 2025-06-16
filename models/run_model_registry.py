# main.py
from jet.models.model_registry.base import ModelFeatures
from jet.models.model_registry.transformers.bert_model_registry import BERTModelRegistry
from jet.models.model_registry.non_transformers.xgboost_model_registry import XGBoostModelRegistry


def main():
    # Transformer-based model
    bert_registry = BERTModelRegistry()
    features: ModelFeatures = {"precision": "fp32"}  # Auto-detect MPS
    bert_model = bert_registry.load_model("bert-base-uncased", features)
    tokenizer = bert_registry.get_tokenizer("bert-base-uncased")
    inputs = tokenizer(["Test sentence"], return_tensors="pt")
    if hasattr(bert_model, "get_embeddings"):
        embeddings = bert_model.get_embeddings(["Test sentence"])
    else:
        embeddings = bert_model(**inputs).last_hidden_state.mean(dim=1)
    print(f"BERT embeddings: {embeddings.shape}")

    # Non-transformer model
    xgboost_registry = XGBoostModelRegistry()
    xgb_model = xgboost_registry.load_model("xgboost_model.pkl")
    import numpy as np
    X = np.array([[1, 2, 3]])
    predictions = xgb_model.predict(xgb.DMatrix(X))
    print(f"XGBoost predictions: {predictions}")


if __name__ == "__main__":
    main()

from jet.models.model_registry.transformers.cross_encoder_model_registry import CrossEncoderRegistry

if __name__ == '__main__':
    model = CrossEncoderRegistry.load_model(
        'cross-encoder/ms-marco-MiniLM-L6-v2')

    candidates = [
        "numpy.linalg.linalg",
        "numpy.core.multiarray",
        "pandas.core.frame",
        "matplotlib.pyplot",
        "sklearn.linear_model",
        "torch.nn.functional",
    ]

    queries = [
        "import matplotlib.pyplot as plt",
        "from numpy.linalg import inv",
        "import torch"
    ]

    # Create pairs of each query with each module path
    pairs = [(query, path) for query in queries for path in candidates]

    # Compute relevance scores
    scores = model.predict(pairs)
    ranked_results = sorted(zip(candidates, scores),
                            key=lambda x: x[1], reverse=True)
    print("Results:", ranked_results)

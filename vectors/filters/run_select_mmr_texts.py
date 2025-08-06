import os
import numpy as np

from jet.file.utils import save_file
from jet.vectors.filters.mmr import select_mmr_texts

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

if __name__ == "__main__":
    # Real-world example: Document snippets about machine learning
    texts = [
        "Machine learning is a method of data analysis that automates model building.",
        "Deep learning, a subset of machine learning, uses neural networks for complex tasks.",
        "Supervised learning involves training models on labeled datasets.",
        "Unsupervised learning finds patterns in unlabeled data using clustering.",
        "Python is a popular programming language for machine learning development.",
        "Reinforcement learning optimizes decisions through trial and error.",
        "Data preprocessing is critical for effective machine learning models."
    ]

    # Simulated embeddings (simplified 3D vectors for demonstration)
    # In practice, these would come from a model like BERT or SentenceTransformer
    embeddings = np.array([
        [0.8, 0.4, 0.1],  # Machine learning overview
        [0.7, 0.5, 0.2],  # Deep learning
        [0.6, 0.3, 0.4],  # Supervised learning
        [0.5, 0.2, 0.5],  # Unsupervised learning
        [0.2, 0.9, 0.3],  # Python programming
        [0.4, 0.3, 0.6],  # Reinforcement learning
        [0.7, 0.4, 0.2]   # Data preprocessing
    ])

    # Simulated query embedding for "machine learning"
    query_embedding = np.array([0.8, 0.4, 0.2])

    # IDs for tracking (optional, will be generated if not provided)
    ids = [f"doc_{i}" for i in range(len(texts))]

    # Run MMR with a balance of relevance and diversity (lambda=0.5)
    results = select_mmr_texts(
        embeddings=embeddings,
        texts=texts,
        query_embedding=query_embedding,
        lambda_param=0.5,
        max_texts=4,
        ids=ids
    )

    # Print results
    print("Selected diverse texts:")
    for result in results:
        print(
            f"ID: {result['id']}, Index: {result['index']}, Score: {result['score']:.4f}")
        print(f"Text: {result['text']}\n")

    save_file(results, f"{OUTPUT_DIR}/results.json")

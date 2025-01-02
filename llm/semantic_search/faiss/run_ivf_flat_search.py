import numpy as np
import faiss


def main_generate_embeddings():
    """
    Generates sample embeddings for the database (products) and queries (user browsing history).
    """
    d = 64  # dimensionality of embeddings
    nb = 100000  # number of product embeddings
    nq = 10000  # number of query embeddings

    np.random.seed(1234)  # reproducibility
    xb = np.random.random((nb, d)).astype('float32')  # product embeddings
    xb[:, 0] += np.arange(nb) / 1000.0  # add some structure
    xq = np.random.random((nq, d)).astype('float32')  # query embeddings
    xq[:, 0] += np.arange(nq) / 1000.0  # add some structure

    return xb, xq


def main_train_and_search(xb, xq):
    """
    Trains the FAISS index and performs nearest neighbor search.
    """
    d = xb.shape[1]  # dimensionality of embeddings
    nlist = 100  # number of clusters
    k = 4  # number of neighbors to retrieve

    # Initialize the index
    quantizer = faiss.IndexFlatL2(d)  # quantizer for clustering
    index = faiss.IndexIVFFlat(
        quantizer, d, nlist, faiss.METRIC_L2)  # IVF index

    # Train the index
    assert not index.is_trained
    index.train(xb)
    assert index.is_trained

    # Add the product embeddings
    index.add(xb)

    # Perform search
    D, I = index.search(xq, k)
    return D, I


def main_recommend_products(I, queries):
    """
    Maps query results to human-readable product recommendations.
    """
    recommendations = {
        0: ["High-performance Laptop", "Affordable Tablet", "Laptop Bag", "Wireless Mouse"],
        1: ["Adjustable Dumbbells", "Yoga Mat", "Resistance Bands", "Workout Shoes"],
        2: ["Running Shoes (Men)", "Trail Sneakers", "Socks for Runners", "Compression Leggings"],
    }

    for i, query in enumerate(queries):
        print(f"Query {i+1}: {query}")
        print("Recommended Products:")
        for idx in I[i]:
            if idx < len(recommendations):
                print(f"  - {recommendations[idx][0]}")
        print()


def main():
    """
    Main function to tie all the components together.
    """
    # Generate embeddings
    xb, xq = main_generate_embeddings()

    # Train and search
    D, I = main_train_and_search(xb, xq)

    # Define queries for demonstration
    queries = [
        "User viewed laptops and tablets.",
        "User searched for fitness-related items.",
        "User viewed sneakers and running shoes."
    ]

    # Recommend products
    main_recommend_products(I, queries)


if __name__ == "__main__":
    main()

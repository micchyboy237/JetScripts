from jet.llm import VectorSemanticSearch
from jet.logger import logger

if __name__ == "__main__":
    # Sample list of package module paths
    module_paths = [
        "numpy.linalg.linalg",
        "numpy.core.multiarray",
        "pandas.core.frame",
        "matplotlib.pyplot",
        "sklearn.linear_model",
        "torch.nn.functional",
    ]

    # Multiline import argument (query)
    query = """
    from sklearn.linear_model import LogisticRegression
    from numpy.linalg import inv
    """

    # Initialize the VectorSemanticSearch class
    search = VectorSemanticSearch(module_paths)

    # Perform and print the results of each search method
    logger.info("Vector-Based Search:")
    vector_results = search.vector_based_search(query)
    for path, score in vector_results:
        logger.log(f"{path}:", f"{score:.4f}", colors=["DEBUG", "SUCCESS"])

    logger.info("\nBM25 Search:")
    bm25_results = search.bm25_search(query)
    for path, score in bm25_results:
        logger.log(f"{path}:", f"{score:.4f}", colors=["DEBUG", "SUCCESS"])

    logger.info("\nGraph-Based Search:")
    graph_results = search.graph_based_search(query)
    for path, score in graph_results:
        logger.log(f"{path}:", f"{score:.4f}", colors=["DEBUG", "SUCCESS"])

    logger.info("\nCross-Encoder Search:")
    cross_encoder_results = search.cross_encoder_search(query)
    for path, score in cross_encoder_results:
        logger.log(f"{path}:", f"{score:.4f}", colors=["DEBUG", "SUCCESS"])

    logger.info("\nFAISS Search:")
    faiss_results = search.faiss_search(query)
    for path, distance in faiss_results:
        logger.log(f"{path}:", f"{distance:.4f}", colors=["DEBUG", "SUCCESS"])

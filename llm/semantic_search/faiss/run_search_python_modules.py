import faiss  # To prevent error on multiprocessing
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
    query = "import matplotlib.pyplot as plt\nfrom numpy.linalg import inv\nimport torch"

    # Initialize the VectorSemanticSearch class
    search = VectorSemanticSearch(module_paths)

    logger.info("\nQuery:")
    logger.debug(query)

    # Perform and print the results of each search method

    logger.info("\nFAISS Search:")
    faiss_results = search.faiss_search(query)
    for path, score in faiss_results:
        logger.log(f"{path}:", f"{score:.4f}", colors=["DEBUG", "SUCCESS"])

    logger.info("\nVector-Based Search:")
    vector_results = search.vector_based_search(query)
    for path, score in vector_results:
        logger.log(f"{path}:", f"{score:.4f}", colors=["DEBUG", "SUCCESS"])

    logger.info("\nGraph-Based Search:")
    graph_results = search.graph_based_search(query)
    for path, score in graph_results:
        logger.log(f"{path}:", f"{score:.4f}", colors=["DEBUG", "SUCCESS"])

    logger.info("\nCross-Encoder Search:")
    cross_encoder_results = search.cross_encoder_search(query)
    for path, score in cross_encoder_results:
        logger.log(f"{path}:", f"{score:.4f}", colors=["DEBUG", "SUCCESS"])

    logger.info("\nBM25 Search:")
    rerank_results = search.rerank_search(query)
    for path, score in rerank_results:
        logger.log(f"{path}:", f"{score:.4f}", colors=["DEBUG", "SUCCESS"])

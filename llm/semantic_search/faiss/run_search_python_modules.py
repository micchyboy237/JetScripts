import faiss  # To prevent error on multiprocessing
from jet.llm import VectorSemanticSearch
from jet.logger import logger

if __name__ == "__main__":
    # Initialize the VectorSemanticSearch class
    module_paths = [
        "numpy.linalg.linalg",
        "numpy.core.multiarray",
        "pandas.core.frame",
        "matplotlib.pyplot",
        "sklearn.linear_model",
        "torch.nn.functional",
    ]

    query = "import matplotlib.pyplot as plt\nfrom numpy.linalg import inv\nimport torch"

    search = VectorSemanticSearch(module_paths)

    # Perform FAISS search
    faiss_results = search.faiss_search(query)
    logger.info("\nFAISS Search Results:")
    for query_line, group in faiss_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['text']}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

     # Perform Cross-encoder search
    cross_encoder_results = search.cross_encoder_search(query)
    logger.info("\nCross-Encoder Search Results:")
    for query_line, group in cross_encoder_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['text']}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Perform Rerank search
    rerank_results = search.rerank_search(query)
    logger.info("\nRerank Search Results:")
    for query_line, group in rerank_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['document']}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Perform Vector-based search
    vector_results = search.vector_based_search(query)
    logger.info("\nVector-Based Search Results:")
    for query_line, group in vector_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['text']}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Perform Graph-based search
    graph_results = search.graph_based_search(query)
    logger.info("\nGraph-Based Search Results:")
    for query_line, group in graph_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['text']}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

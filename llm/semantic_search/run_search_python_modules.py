import faiss  # To prevent error on multiprocessing
from jet.llm import VectorSemanticSearch
from jet.logger import logger

if __name__ == "__main__":
    # module_paths = [
    #     "numpy.linalg.linalg",
    #     "numpy.core.multiarray",
    #     "pandas.core.frame",
    #     "matplotlib.pyplot",
    #     "sklearn.linear_model",
    #     "torch.nn.functional",
    # ]
    # query = "import matplotlib.pyplot as plt\nfrom numpy.linalg import inv\nimport torch"
    # queries = query.splitlines()
    module_paths = [
        "MATCH path1=((person:Person)-[:WORKED_AT]->(company:Company))",
        "MATCH path2=((company:Company)-[:OWNS]->(project:Project))",
        "MATCH path3=((project:Project)-[:USES]->(technology:Technology))",
        "MATCH path4=((person)-[:WORKED_ON]->(project:Project))",
        "MATCH path5=((person)-[:KNOWS]->(technology:Technology))",
        "MATCH path6=((person)-[:HAS_PORTFOLIO]->(port:Portfolio_Link))",
        "MATCH path7=((project:Project)-[:PUBLISHED_AT]->(port:Portfolio_Link))",
        "MATCH path8=((person)-[:STUDIED_AT]->(education:Education))",
        "MATCH path9=((person)-[:SPEAKS]->(language:Language))",
        "MATCH path10=((person)-[:HAS_CONTACT_INFO]->(contact:Contact))",
        "MATCH path11=((person:Person)-[:HAS_RECENT_INFO]->(recent:Recent))",
    ]
    query = "Tell me about yourself."
    queries = query.splitlines()

    search = VectorSemanticSearch(module_paths)

    # Perform Fusion search
    fusion_results = search.fusion_search(queries)
    logger.info("\nFusion Search Results:")
    for result in fusion_results:
        logger.log(f"{result['text']}:", f"{
            result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Perform FAISS search
    faiss_results = search.faiss_search(queries)
    logger.info("\nFAISS Search Results:")
    for query_line, group in faiss_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['text']}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

     # Perform Cross-encoder search
    cross_encoder_results = search.cross_encoder_search(queries)
    logger.info("\nCross-Encoder Search Results:")
    for query_line, group in cross_encoder_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['text']}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Perform Rerank search
    rerank_results = search.rerank_search(queries)
    logger.info("\nRerank Search Results:")
    for query_line, group in rerank_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['document']}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Perform Vector-based search
    vector_results = search.vector_based_search(queries)
    logger.info("\nVector-Based Search Results:")
    for query_line, group in vector_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['text']}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Perform Graph-based search
    graph_results = search.graph_based_search(queries)
    logger.info("\nGraph-Based Search Results:")
    for query_line, group in graph_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['text']}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

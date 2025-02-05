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
        "MATCH path1=((user:User)-[:HAS_PERSONAL_INFO]->(personal:Personal))",
        "MATCH path2=((user:User)-[:WORKED_AT]->(company:Company))",
        "MATCH path3=((company:Company)-[:OWNS]->(project:Project))",
        "MATCH path4=((project:Project)-[:USES]->(technology:Technology))",
        "MATCH path5=((user)-[:WORKED_ON]->(project:Project))",
        "MATCH path6=((user)-[:KNOWS]->(technology:Technology))",
        "MATCH path7=((user)-[:HAS_PORTFOLIO]->(port:Portfolio_Link))",
        "MATCH path8=((project:Project)-[:PUBLISHED_AT]->(port:Portfolio_Link))",
        "MATCH path9=((user)-[:STUDIED_AT]->(education:Education))",
        "MATCH path10=((user)-[:SPEAKS]->(language:Language))",
        "MATCH path11=((user)-[:HAS_CONTACT_INFO]->(contact:Contact))",
        "MATCH path12=((user:User)-[:HAS_RECENT_INFO]->(recent:Recent))",
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

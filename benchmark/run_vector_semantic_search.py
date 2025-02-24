# Define queries and candidates
from jet.actions.vector_semantic_search import VectorSemanticSearch
from jet.logger import logger


queries = [
    "Used technologies are: Magento, front-end architecture, HTML, CSS, JavaScript, front-end technologies, jQuery, Git, React, Vue.js"
]
candidates = [
    "React.js",
    "React Native",
    "Node.js",
    "Python",
    "PostgreSQL",
    "MongoDB",
    "Firebase",
    "AWS",
]


if __name__ == "__main__":
    search = VectorSemanticSearch(candidates)

    # Perform Vector-based search
    vector_results = search.vector_based_search(queries)
    logger.newline()
    logger.orange("Vector-Based Search Results:")
    for query_idx, (query_line, group) in enumerate(vector_results.items()):
        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # Perform Fusion search
    fusion_results = search.fusion_search(queries)
    logger.newline()
    logger.orange(f"Fusion Search Results ({len(fusion_results)}):")
    for query_idx, (query_line, group) in enumerate(fusion_results.items()):
        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # Perform FAISS search
    faiss_results = search.faiss_search(queries)
    logger.newline()
    logger.orange("FAISS Search Results:")
    for query_idx, (query_line, group) in enumerate(faiss_results.items()):
        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # Perform Graph-based search
    graph_results = search.graph_based_search(queries)
    logger.newline()
    logger.orange("Graph-Based Search Results:")
    for query_idx, (query_line, group) in enumerate(graph_results.items()):
        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # Perform Annoy search
    annoy_results = search.annoy_search(queries)
    logger.newline()
    logger.orange(f"Annoy Search Results ({len(annoy_results)}):")
    for query_idx, (query_line, group) in enumerate(annoy_results.items()):
        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # Perform BM25 search
    bm25_results = search.bm25_search(queries)
    logger.newline()
    logger.orange(f"BM25 Search Results ({len(bm25_results)}):")
    for query_idx, (query_line, group) in enumerate(bm25_results.items()):
        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # Perform Cross-encoder search
    cross_encoder_results = search.cross_encoder_search(queries)
    logger.newline()
    logger.orange("Cross-Encoder Search Results:")
    for query_idx, (query_line, group) in enumerate(cross_encoder_results.items()):
        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

    # Perform Rerank search
    rerank_results = search.rerank_search(queries)
    logger.newline()
    logger.orange("Rerank Search Results:")
    for query_idx, (query_line, group) in enumerate(rerank_results.items()):
        logger.newline()
        logger.log(" -", f"Query {query_idx}:",
                   query_line, colors=["GRAY", "GRAY", "DEBUG"])
        for result in group:
            logger.log("  +", f"{result['text'][:25]}:", f"{
                       result['score']:.4f}", colors=["GRAY", "WHITE", "SUCCESS"])

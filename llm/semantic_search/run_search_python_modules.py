import os
from typing import Any, Optional
import faiss  # To prevent error on multiprocessing
from jet.llm import VectorSemanticSearch
from jet.memory.memgraph import initialize_graph
from jet.memory.utils import combine_paths
from jet.transformers import format_json
from jet.logger import logger
from jet.logger.timer import time_it

# Environment variables for Memgraph connection
MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "")

GRAPH_TEMPLATE = """
{query}
RETURN path;
""".strip()


def setup_graph():
    return initialize_graph(MEMGRAPH_URI, MEMGRAPH_USERNAME, MEMGRAPH_PASSWORD)


graph = setup_graph()


def query_graph(query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
    updated_query = GRAPH_TEMPLATE.format(query=query)
    graph_query_result = graph.query(updated_query)[:top_k]
    return graph_query_result


def call_graph_queries(queries: list[str]) -> list[dict[str, Any]]:
    graph_query_results = []
    for query in queries:
        graph_query_result = query_graph(query)
        graph_query_results.extend(graph_query_result)
    return graph_query_results


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
        "MATCH path=((user:User)-[:HAS_PERSONAL_INFO]->(personal:Personal))",
        "MATCH path=((personal:Personal)-[:WORKED_AT]->(company:Company))",
        "MATCH path=((company:Company)-[:OWNS]->(project:Project))",
        "MATCH path=((project:Project)-[:USES]->(technology:Technology))",
        "MATCH path=((personal)-[:WORKED_ON]->(project:Project))",
        "MATCH path=((personal)-[:KNOWS]->(technology:Technology))",
        "MATCH path=((personal)-[:HAS_PORTFOLIO]->(port:Portfolio_Link))",
        "MATCH path=((project:Project)-[:PUBLISHED_AT]->(port:Portfolio_Link))",
        "MATCH path=((personal)-[:STUDIED_AT]->(education:Education))",
        "MATCH path=((personal)-[:SPEAKS]->(language:Language))",
        "MATCH path=((personal)-[:HAS_CONTACT_INFO]->(contact:Contact))",
        "MATCH path=((personal:Personal)-[:HAS_RECENT_INFO]->(recent:Recent))",
    ]
    graph_query_results = call_graph_queries(module_paths)

    combined_paths = combine_paths(graph_query_results)
    logger.newline()
    logger.debug("combined_paths result:")
    logger.success(format_json(combined_paths))

    transformed_unique_paths = []
    for item in combined_paths:
        source_obj = item[0]
        target_objs = item[2]

        if source_obj not in transformed_unique_paths:
            transformed_unique_paths.append(source_obj)
        for target_obj in target_objs:
            if target_obj not in transformed_unique_paths:
                transformed_unique_paths.append(target_obj)

    module_paths = [str(item) for item in combined_paths]

    query = "Tell me about yourself."
    queries = query.splitlines()

    search = VectorSemanticSearch(module_paths)

    # Perform Fusion search
    fusion_results = search.fusion_search(queries)
    logger.info(f"\nFusion Search Results ({len(fusion_results)}):")
    for result in fusion_results:
        logger.log(f"{result['text'][:50]}:", f"{
            result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Perform FAISS search
    faiss_results = search.faiss_search(queries)
    logger.info("\nFAISS Search Results:")
    for query_line, group in faiss_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['text'][:50]}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

     # Perform Cross-encoder search
    cross_encoder_results = search.cross_encoder_search(queries)
    logger.info("\nCross-Encoder Search Results:")
    for query_line, group in cross_encoder_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['text'][:50]}:", f"{
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
            logger.log(f"{result['text'][:50]}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Perform Graph-based search
    graph_results = search.graph_based_search(queries)
    logger.info("\nGraph-Based Search Results:")
    for query_line, group in graph_results.items():
        logger.info(f"\nQuery line: {query_line}")
        for result in group:
            logger.log(f"{result['text'][:50]}:", f"{
                       result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

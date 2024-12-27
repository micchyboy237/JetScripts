import json
from jet.search import search_searxng
from jet.logger import logger

if __name__ == "__main__":
    from jet.llm.query import setup_index, query_llm, FUSION_MODES
    from jet.data import generate_unique_hash
    from script_utils import display_source_nodes
    from llama_index.core.schema import Document as LlamaDocument

    query = "best pizza places in New York"

    results = search_searxng(
        query_url="http://searxng.local:8080/search",
        query=query,
        min_score=0,
        engines=["google"],
        # use_cache=False,
    )
    logger.log("Search Results:", len(results), colors=["WHITE", "SUCCESS"])

    documents = [LlamaDocument(text=result['content']) for result in results]

    logger.info("Setup index...")
    query_nodes = setup_index(documents)

    # logger.newline()
    # logger.info("RECIPROCAL_RANK: query...")
    # response = query_nodes(sample_query, FUSION_MODES.RECIPROCAL_RANK)

    # logger.newline()
    # logger.info("DIST_BASED_SCORE: query...")
    # response = query_nodes(sample_query, FUSION_MODES.DIST_BASED_SCORE)

    logger.newline()
    logger.info("RELATIVE_SCORE query...")
    result = query_nodes(
        query, FUSION_MODES.RELATIVE_SCORE)
    logger.info(f"RETRIEVED NODES ({len(result["nodes"])})")
    display_source_nodes(query, result["nodes"], source_length=None)

    # response = query_llm(query, result['texts'])
    response = result['texts'][0]
    logger.info("search_web_results response:")
    logger.success(response)

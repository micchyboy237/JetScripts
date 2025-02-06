import json
import os
import re
from typing import Any, Optional
import faiss  # To prevent error on multiprocessing
from jet.llm import VectorSemanticSearch
from jet.llm.query.retrievers import query_llm
from jet.memory.memgraph import initialize_graph, query_memgraph
from jet.memory.memgraph_types import GraphQueryMetadata, GraphQueryRequest
from jet.memory.utils import combine_paths
from jet.transformers import format_json
from jet.logger import logger
from jet.logger.timer import time_it
from jet.transformers.object import make_serializable

# Environment variables for Memgraph connection
MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "")

GRAPH_TEMPLATE = """
{query}
RETURN source, type(action) AS action, COLLECT(target) AS targets;
""".strip()

GRAPH_BY_ID_TEMPLATE = """
MATCH (path)
WHERE path.__mg_id__ = {id}
RETURN path, path.__mg_id__, labels(path) AS path_labels;
""".strip()


def setup_graph():
    return initialize_graph(MEMGRAPH_URI, MEMGRAPH_USERNAME, MEMGRAPH_PASSWORD)


graph = setup_graph()

metadata = {
    "queryId": "147c4b5a-45bf-41cb-aee1-70a9b1b3055d",
    "source": "lab-user"
}


def query_graph(query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
    # Create an instance of GraphQueryMetadata
    metadata_instance = GraphQueryMetadata(
        queryId=metadata["queryId"],
        source=metadata["source"],
    )
    # Create an instance of GraphQueryRequest using the metadata instance
    request_instance = GraphQueryRequest(
        query=query, metadata=metadata_instance)
    memgraph_results = query_memgraph(request_instance)
    graph_query_results = [item
                           for item in memgraph_results['records']]
    graph_query_results = graph_query_results[:top_k]

    # graph_query_results = graph.query(query)[:top_k]
    return graph_query_results


def call_graph_queries(queries: list[str]) -> list[dict[str, Any]]:
    graph_query_results = []
    for query in queries:
        # Use regex to find the node type inside parentheses
        match = re.search(r'\((\w+)(?::\w+)?\)', query)
        if match:
            node_type = match.group(1)
            updated_query = GRAPH_TEMPLATE.format(
                query=query, type=node_type.lower())

            query_results = query_graph(updated_query)
            graph_query_results.extend(query_results)
    return graph_query_results


def call_graph_queries_by_ids(query_ids: list[int]) -> list[dict[str, Any]]:
    graph_query_results = []
    for id in query_ids:
        updated_query = GRAPH_BY_ID_TEMPLATE.format(id=id)

        graph_query_result = query_graph(updated_query)
        graph_query_results.extend(graph_query_result)
    return graph_query_results


def validate_object_with_descriptive_attributes(obj: dict[str, Any]) -> bool:
    temp_obj = obj.copy()
    temp_obj.pop('__mg_id__')
    return len(temp_obj) > 0


def update_module_paths(unique_combined_paths: list[dict[str, Any]]) -> list[str]:
    updated_paths = []

    for item in unique_combined_paths:
        item = item.copy()
        # Get the '__mg_id__' key value first
        mg_id = item.pop('__mg_id__', None)
        if mg_id is not None and len(item) > 0:
            # Create a new dict starting with '__mg_id__'
            # sorted_item = {'__mg_id__': mg_id}
            sorted_item = {}
            # Sort the rest of the keys in the item and add them to the dictionary
            sorted_item.update({k: item[k] for k in sorted(item.keys())})
            # Convert the sorted item to string
            updated_paths.append(str(sorted_item))

    return updated_paths


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
        "MATCH (source)-[action:HAS_PERSONAL_INFO]->(target)",
        "MATCH (source)-[action:WORKED_AT]->(target)",
        "MATCH (source)-[action:OWNS]->(target)",
        "MATCH (source)-[action:USES]->(target)",
        "MATCH (source)-[action:WORKED_ON]->(target)",
        "MATCH (source)-[action:KNOWS]->(target)",
        "MATCH (source)-[action:HAS_PORTFOLIO]->(target)",
        "MATCH (source)-[action:PUBLISHED_AT]->(target)",
        "MATCH (source)-[action:STUDIED_AT]->(target)",
        "MATCH (source)-[action:SPEAKS]->(target)",
        "MATCH (source)-[action:HAS_CONTACT_INFO]->(target)",
        "MATCH (source)-[action:HAS_RECENT_INFO]->(target)",
    ]
    graph_query_results = call_graph_queries(module_paths)
    combined_paths = combine_paths(graph_query_results)
    # logger.newline()
    # logger.debug("combined_paths result:")
    # logger.success(format_json(combined_paths))

    def filter_unique_œuery_ids(item_paths):
        unique_query_ids = []
        for item in item_paths:
            source_obj = item["source"]
            target_objs = item["targets"]

            if validate_object_with_descriptive_attributes(source_obj) and source_obj['__mg_id__'] not in unique_query_ids:
                unique_query_ids.append(source_obj['__mg_id__'])
            for target_obj in target_objs:
                if validate_object_with_descriptive_attributes(target_obj) and target_obj['__mg_id__'] not in unique_query_ids:
                    unique_query_ids.append(target_obj['__mg_id__'])
        return unique_query_ids

    unique_query_ids = filter_unique_œuery_ids(combined_paths)
    unique_graph_query_results_by_ids = [
        item['path'] for item in call_graph_queries_by_ids(unique_query_ids)]
    combined_paths_by_ids = combine_paths(unique_graph_query_results_by_ids)
    graph_query_results_by_ids_dict = {
        item['__mg_id__']: item for item in combined_paths_by_ids}

    logger.newline()
    logger.debug("combined_paths_by_ids result:")
    logger.success(format_json(combined_paths_by_ids))

    # Generate llm chat context
    def generate_query_candidates(graph_results):
        candidates = []
        for item in graph_results:
            candidates.append(json.dumps(make_serializable(item)))

        return candidates

    def generate_query_contexts(graph_results):
        context_items = []
        for item in graph_results:
            source_id = item["source"]['__mg_id__']
            source_label = item["source"]['_label']
            source_str = f"{str(source_id)}:{source_label}"

            action = item['action'].replace("_", " ").title()

            targets = []
            for target in item["targets"]:
                target_id = target['__mg_id__']
                target_label = target['_label']
                target_str = f"{str(target_id)}:{target_label}"

                targets.append(target_str)
            targets_str = ", ".join(targets)

            item_contexts = [
                f"Source: {source_str}",
                f"Action: {action}",
                f"Targets: [{targets_str}]",
            ]
            item_contexts_str = " | ".join(item_contexts)
            context_items.append(item_contexts_str)

        return context_items

    module_paths = generate_query_candidates(combined_paths)

    query = "Tell me about yourself."
    queries = query.splitlines()

    search = VectorSemanticSearch(module_paths)

    # Perform Fusion search
    fusion_results = search.fusion_search(queries)
    logger.info(f"\nFusion Search Results ({len(fusion_results)}):")
    for result in fusion_results:
        logger.log(f"{result['text'][:50]}:", f"{
            result['score']:.4f}", colors=["DEBUG", "SUCCESS"])

    # Query LLM with Fusion contexts
    fusion_contexts = []
    for result in fusion_results:
        try:
            result_dict = json.loads(result['text'])
            fusion_contexts.append(result_dict)
        except:
            continue

    unique_query_ids = filter_unique_œuery_ids(fusion_contexts)
    unique_graph_query_results_by_ids = [
        item['path'] for item in call_graph_queries_by_ids(unique_query_ids)]
    combined_paths_by_ids = combine_paths(unique_graph_query_results_by_ids)
    graph_query_results_by_ids_dict = {
        item['__mg_id__']: item for item in combined_paths_by_ids}

    llm_contexts = generate_query_contexts(fusion_contexts)

    def filter_context_value(value: dict):
        filtered_value = {k: v for k,
                          v in value.items() if not k.startswith("_")}
        return filtered_value

    context_values = [
        str({
            '__mg_id__': f"{value['__mg_id__']}:{value['_label']}", **filter_context_value(value)
        }) for value in graph_query_results_by_ids_dict.values()
    ]

    contexts = [*context_values, *llm_contexts]
    response = query_llm(query, contexts)
    logger.newline()
    logger.debug("Fusion query response:")
    logger.success(response)

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

# Setup LLM settings
import json
import os

from tqdm import tqdm
from jet.memory.config import CONTEXT_DB_TEMPLATE, CONTEXT_SAMPLES_TEMPLATE, CONTEXT_SCHEMA_TEMPLATE
from jet.memory.memgraph import generate_query, generate_cypher_query, initialize_graph
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.file import load_file


MODEL = "llama3.2"

# Setup Memgraph variables
URL = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
USERNAME = os.environ.get("MEMGRAPH_USERNAME", "")
PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "")

# Function to log the result


def log_result(query: str, result: str):
    logger.newline()
    logger.debug(query)
    logger.success(result)


# Main function to execute the process
def main():
    # cypher_queries_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/memgraph/data/jet-resume/db.cypherl"
    # data_query = load_file(cypher_queries_file)
    data_query = None

    sample_queries_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/workflows/prompt_enhancer/memory/sample_queries.cypherl"
    sample_queries_str = load_file(sample_queries_file)

    # Initialize Memgraph
    graph = initialize_graph(URL, USERNAME, PASSWORD, data_query)

    # Query graph (you can adjust the query as per your needs)
    query = "Tell me about yourself."
    tone_name = "an employer"
    num_of_queries = 3

    # Generate cypher query
    generated_cypher_queries = generate_cypher_query(
        query, graph, tone_name, num_of_queries=num_of_queries, samples=sample_queries_str)

    used_cypher_queries = []
    graph_result_contexts = []
    for idx, cypher_query in enumerate(generated_cypher_queries):
        log_result(query, cypher_query)

        # Query the graph
        top_k = None
        graph_result = graph.query(cypher_query)[:top_k]

        if graph_result:
            logger.newline()
            logger.info(f"Graph Result {idx + 1}:")
            logger.success(graph_result)

            used_cypher_queries.append(cypher_query)
            graph_result_contexts.append(json.dumps(graph_result))

    # Generate query results
    db_results = []
    for item, result in zip(used_cypher_queries, graph_result_contexts):
        db_results.append(f"Query: {item}\nResult: {result}")

    db_results_str = CONTEXT_DB_TEMPLATE.format(
        db_results_str="\n\n".join(db_results))

    schema_str = CONTEXT_SCHEMA_TEMPLATE.format(
        schema_str=graph.get_schema)

    contexts = [
        db_results_str,
        schema_str
    ]
    context = "\n\n".join(contexts)

    tone_name = "a professional"
    response = generate_query(query, tone_name, context=context)

    result = ""
    for chunk in response:
        result += chunk

    logger.newline()
    logger.info("Query:")
    logger.debug(query)

    logger.newline()
    logger.info("Query Result:")
    logger.success(result)


if __name__ == "__main__":
    main()

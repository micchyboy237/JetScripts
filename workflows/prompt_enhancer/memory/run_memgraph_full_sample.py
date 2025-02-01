# Setup LLM settings
import os
from jet.memory.memgraph import generate_context_and_query, generate_cypher_query, initialize_graph
from jet.logger import logger
from jet.transformers import format_json


MODEL = "llama3.2"

# Setup Memgraph variables
URL = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
USERNAME = os.environ.get("MEMGRAPH_USERNAME", "")
PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "")

# Function to log the result


def log_result(query: str, result: str):
    logger.newline()
    logger.debug(query)
    logger.success(format_json(result))


# Main function to execute the process
def main():
    data_query = """
        MERGE (g:Game {name: "Baldur's Gate 3"})
        WITH g, ["PlayStation 5", "Mac OS", "Windows", "Xbox Series X/S"] AS platforms,
                ["Adventure", "Role-Playing Game", "Strategy"] AS genres
        FOREACH (platform IN platforms |
            MERGE (p:Platform {name: platform})
            MERGE (g)-[:AVAILABLE_ON]->(p)
        )
        FOREACH (genre IN genres |
            MERGE (gn:Genre {name: genre})
            MERGE (g)-[:HAS_GENRE]->(gn)
        )
        MERGE (p:Publisher {name: "Larian Studios"})
        MERGE (g)-[:PUBLISHED_BY]->(p);
    """
    # Initialize Memgraph
    graph = initialize_graph(URL, USERNAME, PASSWORD, data_query)

    # Query graph (you can adjust the query as per your needs)
    query = "Is Baldur's Gate 3 available on PS5?"

    # Generate cypher query
    generated_cypher = generate_cypher_query(query, graph)
    log_result(query, generated_cypher)

    # Query the graph
    top_k = None
    graph_result_context = graph.query(generated_cypher)[:top_k]

    logger.newline()
    logger.info("Graph Result Context:")
    logger.success(graph_result_context)

    # Generate context and query
    result = generate_context_and_query(
        generated_cypher, query, graph_result_context)

    logger.newline()
    logger.info("Query:")
    logger.debug(query)

    logger.newline()
    logger.info("Query Result:")
    logger.success(result)


if __name__ == "__main__":
    main()

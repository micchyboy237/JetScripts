import os
from jet.validation.cypher_graph_validator import validate_query
from jet.logger import logger


url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
username = os.environ.get("MEMGRAPH_USERNAME", "")
password = os.environ.get("MEMGRAPH_PASSWORD", "")


def main_correct_query():
    query = "MATCH (g:Game {name: \"Baldur's Gate 3\"})-[:AVAILABLE_ON]->(p:Platform)\nRETURN p.name"
    validation_response = validate_query(query)

    logger.newline()
    logger.info("Graph Result:")
    logger.success(validation_response)


def main_incorrect_query():
    query = "MATCH (g:Game)-[:AVAILABLE_ON]->(p:Platform) WHERE g.name = 'Baldur''s Gate 3' AND p.name = 'PS5' RETURN g, p"
    validation_response = validate_query(query)

    logger.newline()
    logger.info("Graph Result:")
    logger.success(validation_response)


if __name__ == "__main__":
    main_correct_query()
    main_incorrect_query()

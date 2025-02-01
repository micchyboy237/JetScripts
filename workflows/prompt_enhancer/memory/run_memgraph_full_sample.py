from typing import Optional
from jet.llm.main.generation import call_ollama_chat
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
import os
from langchain_community.graphs import MemgraphGraph
from langchain_core.prompts import PromptTemplate
from jet.transformers import format_json
from config import (
    CYPHER_GENERATION_TEMPLATE,
    CYPHER_GENERATION_PROMPT,
    CONTEXT_QA_PROMPT,
    CONTEXT_PROMPT_TEMPLATE,
)

initialize_ollama_settings()

# Setup LLM settings
MODEL = "llama3.2"

# Setup Memgraph variables
URL = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
USERNAME = os.environ.get("MEMGRAPH_USERNAME", "")
PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "")

# Function to initialize Memgraph


def initialize_graph(url: str, username: str, password: str, data_query: Optional[str] = None) -> MemgraphGraph:
    graph = MemgraphGraph(url=url, username=username,
                          password=password, refresh_schema=False)
    if data_query:
        graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
        graph.query("DROP GRAPH")
        graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")

        graph.query(data_query)

        graph.refresh_schema()

    return graph

# Function to log the result


def log_result(query: str, result: str):
    logger.newline()
    logger.debug(query)
    logger.success(format_json(result))

# Function to generate cypher query


def generate_cypher_query(query: str, graph: MemgraphGraph) -> str:
    cypher_generation_query = CYPHER_GENERATION_TEMPLATE.format(
        query_str=query)
    prompt = CYPHER_GENERATION_PROMPT.format(
        schema=graph.get_schema,
        prompt=cypher_generation_query,
    )
    generated_cypher = ""
    for chunk in call_ollama_chat(
        prompt,
        stream=True,
        model=MODEL,
        options={"seed": 42, "temperature": 0,
                 "num_keep": 0, "num_predict": -1},
    ):
        generated_cypher += chunk
    return generated_cypher

# Function to generate context and query


def generate_context_and_query(cypher: str, query: str, graph_result_context: str) -> str:
    context = CONTEXT_PROMPT_TEMPLATE.format(
        cypher_query_str=cypher.strip('"'),
        graph_result_str=graph_result_context,
    )
    prompt = CONTEXT_QA_PROMPT.format(context=context, question=query)
    result = ""
    for chunk in call_ollama_chat(
        prompt,
        stream=True,
        model=MODEL,
        options={"seed": 42, "temperature": 0,
                 "num_keep": 0, "num_predict": -1},
    ):
        result += chunk
    return result

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

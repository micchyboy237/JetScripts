from jet.llm.main.generation import call_ollama_chat
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
import os
from langchain_community.graphs import MemgraphGraph
from langchain_core.prompts import PromptTemplate
from jet.transformers import format_json

initialize_ollama_settings()

# Setup LLM settings

model = "llama3.2"


# Setup variables

url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
username = os.environ.get("MEMGRAPH_USERNAME", "")
password = os.environ.get("MEMGRAPH_PASSWORD", "")

graph = MemgraphGraph(
    url=url, username=username, password=password, refresh_schema=False
)

graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
graph.query("DROP GRAPH")
graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")

query = """
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

result = graph.query(query)

logger.newline()
logger.debug(query)
logger.success(format_json(result))


# Refresh schema

graph.refresh_schema()


# Query graph

MEMGRAPH_GENERATION_TEMPLATE = """
Your task is to directly translate natural language inquiry into precise and executable Cypher query for Memgraph database. 
You will utilize a provided database schema to understand the structure, nodes and relationships within the Memgraph database.
Instructions: 
- Use provided node and relationship labels and property names from the
schema which describes the database's structure. Upon receiving a user
question, synthesize the schema to craft a precise Cypher query that
directly corresponds to the user's intent. 
- Generate valid executable Cypher queries on top of Memgraph database. 
Any explanation, context, or additional information that is not a part 
of the Cypher query syntax should be omitted entirely. 
- Use Memgraph MAGE procedures instead of Neo4j APOC procedures. 
- Do not include any explanations or apologies in your responses. 
- Do not include any text except the generated Cypher statement.
- For queries that ask for information or functionalities outside the direct
generation of Cypher queries, use the Cypher query format to communicate
limitations or capabilities. For example: RETURN "I am designed to generate
Cypher queries based on the provided schema only."
Schema: 
{schema}

With all the above information and instructions, generate Cypher query for the
user prompt. 

The prompt is:
{prompt}
""".strip()

MEMGRAPH_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"], template=MEMGRAPH_GENERATION_TEMPLATE
)

CYPHER_GENERATION_TEMPLATE = """
I want a more generic cypher query without specific filters. Given this prompt "{query_str}", write one.
""".strip()


query = """
MATCH (g:Game {name: "Baldur's Gate 3"})-[:AVAILABLE_ON]->(p:Platform {name: "PlayStation 5"}) RETURN p.name
""".strip()
query = """
Is Baldur's Gate 3 available on PS5?
""".strip()
cypher_generation_query = CYPHER_GENERATION_TEMPLATE.format(
    query_str=query)

prompt = MEMGRAPH_GENERATION_PROMPT.format(**{
    "schema": graph.get_schema,
    "prompt": cypher_generation_query,
})

generated_cypher = ""
for chunk in call_ollama_chat(
    prompt,
    stream=True,
    model=model,
    options={
        "seed": 42,
        "temperature": 0,
        "num_keep": 0,
        "num_predict": -1,
    },
):
    generated_cypher += chunk

logger.newline()
logger.info("Cypher Prompt:")
logger.debug(prompt)

logger.newline()
logger.info("Cypher Result:")
logger.success(generated_cypher)


# Query context

top_k = None
graph_result_context = graph.query(generated_cypher)[: top_k]

logger.newline()
logger.info("Graph Result Context:")
logger.success(graph_result_context)


# Query graph

MEMGRAPH_QA_TEMPLATE = """Your task is to form nice and human understandable answers. The context contains the cypher query result that you must use to construct an answer. The provided context is authoritative, you must never doubt it or try to use your internal knowledge to correct it. Make the answer sound as a response to the question. Do not mention that you based the result on the given context. Here is an example:

Question: Which managers own Neo4j stocks?
Context: [manager:CTL LLC, manager:JANE STREET GROUP LLC]
Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

Follow this example when generating answers. If the provided context is empty, say that you don't know the answer.


Question: {question}
Context: {context}
Helpful Answer:"""

MEMGRAPH_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=MEMGRAPH_QA_TEMPLATE
)

CONTEXT_PROMPT_TEMPLATE = """
Cypher query: {cypher_query_str}
Result: {graph_result_str}
""".strip()


query = "Is Baldur's Gate 3 available on PS5?"

context = CONTEXT_PROMPT_TEMPLATE.format(**{
    "cypher_query_str": generated_cypher,
    "graph_result_str": graph_result_context,
})

prompt = MEMGRAPH_QA_PROMPT.format(**{
    "context": context,
    "question": query,
})

result = ""
for chunk in call_ollama_chat(
    prompt,
    stream=True,
    model=model,
    options={
        "seed": 42,
        "temperature": 0,
        "num_keep": 0,
        "num_predict": -1,
    },
):
    result += chunk

logger.newline()
logger.info("Query:")
logger.debug(query)

logger.newline()
logger.info("Query Result:")
logger.success(result)

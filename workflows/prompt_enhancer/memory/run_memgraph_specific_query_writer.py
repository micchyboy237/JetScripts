from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
import os
from langchain_community.chains.graph_qa.memgraph import MemgraphQAChain
from langchain_community.graphs import MemgraphGraph
from langchain_core.prompts import PromptTemplate
from jet.llm.ollama.base_langchain import ChatOllama
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from jet.transformers import format_json

# initialize_ollama_settings()

model = "llama3.2"
MEMGRAPH_QA_TEMPLATE = """Your task is to form nice and human understandable answers. The context contains the cypher query result that you must use to construct an answer. The provided context is authoritative, you must never doubt it or try to use your internal knowledge to correct it. Make the answer sound as a response to the question. Do not mention that you based the result on the given context. Here is an example:

Question: Which managers own Neo4j stocks?
Context:[manager:CTL LLC, manager:JANE STREET GROUP LLC]
Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

Follow this example when generating answers. If the provided context is empty, say that you don't know the answer.


Question: {question}
Context:{context}
Helpful Answer:"""

MEMGRAPH_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=MEMGRAPH_QA_TEMPLATE
)

MEMGRAPH_GENERATION_TEMPLATE = """Your task is to directly translate natural language inquiry into precise and executable Cypher query for Memgraph database. 
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
{question}"""

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

MEMGRAPH_GENERATION_QUERY = """
I want a more generic query without specific filters. Given this query "{cypher_str}", write a one that shows all the platforms that contains it.
""".strip()

url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
username = os.environ.get("MEMGRAPH_USERNAME", "")
password = os.environ.get("MEMGRAPH_PASSWORD", "")

graph = MemgraphGraph(
    url=url, username=username, password=password, refresh_schema=False
)

chain = MemgraphQAChain.from_llm(
    ChatOllama(model=model),
    graph=graph,
    model_name=model,
    allow_dangerous_requests=True,
)


cypher_str = """
MATCH (g:Game {name: "Baldur's Gate 3"})-[:AVAILABLE_ON]->(p:Platform {name: "PlayStation 5"}) RETURN p.name
""".strip()
query = MEMGRAPH_GENERATION_QUERY.format(cypher_str=cypher_str)

query = "Is Baldur's Gate 3 available on PS5?"
response = chain.invoke(query)

logger.newline()
logger.debug(query)
logger.success(format_json(response["result"]))

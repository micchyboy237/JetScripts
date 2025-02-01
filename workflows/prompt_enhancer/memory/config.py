from langchain_core.prompts import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
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

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"], template=CYPHER_GENERATION_TEMPLATE
)

CYPHER_GENERATION_TEMPLATE = """
I want a more generic cypher query without filters.
Do not include WHERE clauses.
Generate the least code possible.
Write one given this prompt "{query_str}".
""".strip()


CONTEXT_QA_TEMPLATE = """Your task is to form nice and human understandable answers. The context contains the cypher query result that you must use to construct an answer. The provided context is authoritative, you must never doubt it or try to use your internal knowledge to correct it. Make the answer sound as a response to the question. Do not mention that you based the result on the given context. Here is an example:

Question: Which managers own Neo4j stocks?
Context: [manager:CTL LLC, manager:JANE STREET GROUP LLC]
Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

Follow this example when generating answers. If the provided context is empty, say that you don't know the answer.


Question: {question}
Context: {context}
Helpful Answer:"""

CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CONTEXT_QA_TEMPLATE
)

CONTEXT_PROMPT_TEMPLATE = """
Cypher query: {cypher_query_str}
Cypher result:
{graph_result_str}
""".strip()

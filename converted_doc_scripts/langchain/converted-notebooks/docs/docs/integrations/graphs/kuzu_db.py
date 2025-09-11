from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_kuzu.chains.graph_qa.kuzu import KuzuQAChain
from langchain_kuzu.graphs.kuzu_graph import KuzuGraph
import kuzu
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Kuzu

> [Kùzu](https://kuzudb.com/) is an embeddable, scalable, extremely fast graph database.
> It is permissively licensed with an MIT license, and you can see its source code [here](https://github.com/kuzudb/kuzu).

> Key characteristics of Kùzu:
>- Performance and scalability: Implements modern, state-of-the-art join algorithms for graphs.
>- Usability: Very easy to set up and get started with, as there are no servers (embedded architecture).
>- Interoperability: Can conveniently scan and copy data from external columnar formats, CSV, JSON and relational databases.
>- Structured property graph model: Implements the property graph model, with added structure.
>- Cypher support: Allows convenient querying of the graph in Cypher, a declarative query language.

> Get started with Kùzu by visiting their [documentation](https://docs.kuzudb.com/).

## Setting up

Kùzu is an embedded database (it runs in-process), so there are no servers to manage. Install the
following dependencies to get started:

```bash
pip install -U langchain-kuzu langchain-ollama langchain-experimental
```

This installs Kùzu along with the LangChain integration for it, as well as the Ollama Python package
so that we can use Ollama's LLMs. If you want to use other LLM providers, you can install their
respective Python packages that come with LangChain.

Here's how you would first create a Kùzu database on your local machine and connect to it:
"""
logger.info("# Kuzu")


db = kuzu.Database("test_db")
conn = kuzu.Connection(db)

"""
## Create `KuzuGraph`

Kùzu's integration with LangChain makes it convenient to create and update graphs from unstructured text, and also to query graphs via a Text2Cypher pipeline that utilizes the
power of LangChain's LLM chains. To begin, we create a `KuzuGraph` object that uses the database object we created above in combination with the `KuzuGraph` constructor.
"""
logger.info("## Create `KuzuGraph`")


graph = KuzuGraph(db, allow_dangerous_requests=True)

"""
Say we want to transform the following text into a graph:
"""
logger.info("Say we want to transform the following text into a graph:")

text = "Tim Cook is the CEO of Apple. Apple has its headquarters in California."

"""
We will make use of `LLMGraphTransformer` to use an LLM to extract nodes and relationships from the text.
To make the graph more useful, we will define the following schema, such that the LLM will only
extract nodes and relationships that match the schema.
"""
logger.info("We will make use of `LLMGraphTransformer` to use an LLM to extract nodes and relationships from the text.")

allowed_nodes = ["Person", "Company", "Location"]
allowed_relationships = [
    ("Person", "IS_CEO_OF", "Company"),
    ("Company", "HAS_HEADQUARTERS_IN", "Location"),
]

"""
The `LLMGraphTransformer` class provides a convenient way to convert the text into a list of graph documents.
"""
logger.info("The `LLMGraphTransformer` class provides a convenient way to convert the text into a list of graph documents.")


llm_transformer = LLMGraphTransformer(
#     llm=ChatOllama(model="llama3.2"),  # noqa: F821
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships,
)

documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)

graph_documents[:2]

"""
We can then call the above defined `KuzuGraph` object's `add_graph_documents` method to ingest the graph documents into the Kùzu database.
The `include_source` argument is set to `True` so that we also create relationships between each entity node and the source document that it came from.
"""
logger.info("We can then call the above defined `KuzuGraph` object's `add_graph_documents` method to ingest the graph documents into the Kùzu database.")

graph.add_graph_documents(
    graph_documents,
    include_source=True,
)

"""
## Creating `KuzuQAChain`

To query the graph via a Text2Cypher pipeline, we can define a `KuzuQAChain` object. Then, we can invoke the chain with a query by connecting to the existing database that's stored in the `test_db` directory defined above.
"""
logger.info("## Creating `KuzuQAChain`")


chain = KuzuQAChain.from_llm(
#     llm=ChatOllama(model="llama3.2"),  # noqa: F821
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

"""
Note that we set a temperature that's slightly higher than zero to avoid the LLM being overly concise in its response.

Let's ask some questions using the QA chain.
"""
logger.info("Note that we set a temperature that's slightly higher than zero to avoid the LLM being overly concise in its response.")

chain.invoke("Who is the CEO of Apple?")

chain.invoke("Where is Apple headquartered?")

"""
## Refresh graph schema

If you mutate or update the graph, you can inspect the refreshed schema information that's used by the Text2Cypher chain to generate Cypher statements.
You don't need to manually call `refresh_schema()` each time as it's called automatically when you invoke the chain.
"""
logger.info("## Refresh graph schema")

graph.refresh_schema()

logger.debug(graph.get_schema)

"""
## Use separate LLMs for Cypher and answer generation

You can specify `cypher_llm` and `qa_llm` separately to use different LLMs for Cypher generation and answer generation.
"""
logger.info("## Use separate LLMs for Cypher and answer generation")

chain = KuzuQAChain.from_llm(
    cypher_llm=ChatOllama(model="llama3.2"),
    qa_llm=ChatOllama(model="llama3.2"),
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

chain.invoke("Who is the CEO of Apple?")

logger.info("\n\n[DONE]", bright=True)
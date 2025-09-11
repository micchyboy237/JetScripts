from hugegraph.connection import PyHugeGraph
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import HugeGraphQAChain
from langchain_community.graphs import HugeGraph
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
# HugeGraph

>[HugeGraph](https://hugegraph.apache.org/) is a convenient, efficient, and adaptable graph database compatible with
>the `Apache TinkerPop3` framework and the `Gremlin` query language.
>
>[Gremlin](https://en.wikipedia.org/wiki/Gremlin_(query_language)) is a graph traversal language and virtual machine developed by `Apache TinkerPop` of the `Apache Software Foundation`.

This notebook shows how to use LLMs to provide a natural language interface to [HugeGraph](https://hugegraph.apache.org/cn/) database.

## Setting up

You will need to have a running HugeGraph instance.
You can run a local docker container by running the executing the following script:

```
docker run \
    --name=graph \
    -itd \
    -p 8080:8080 \
    hugegraph/hugegraph
```

If we want to connect HugeGraph in the application, we need to install python sdk:

```
pip3 install hugegraph-python
```

If you are using the docker container, you need to wait a couple of second for the database to start, and then we need create schema and write graph data for the database.
"""
logger.info("# HugeGraph")


client = PyHugeGraph("localhost", "8080", user="admin", pwd="admin", graph="hugegraph")

"""
First, we create the schema for a simple movie database:
"""
logger.info("First, we create the schema for a simple movie database:")

"""schema"""
schema = client.schema()
schema.propertyKey("name").asText().ifNotExist().create()
schema.propertyKey("birthDate").asText().ifNotExist().create()
schema.vertexLabel("Person").properties(
    "name", "birthDate"
).usePrimaryKeyId().primaryKeys("name").ifNotExist().create()
schema.vertexLabel("Movie").properties("name").usePrimaryKeyId().primaryKeys(
    "name"
).ifNotExist().create()
schema.edgeLabel("ActedIn").sourceLabel("Person").targetLabel(
    "Movie"
).ifNotExist().create()

"""
Then we can insert some data.
"""
logger.info("Then we can insert some data.")

"""graph"""
g = client.graph()
g.addVertex("Person", {"name": "Al Pacino", "birthDate": "1940-04-25"})
g.addVertex("Person", {"name": "Robert De Niro", "birthDate": "1943-08-17"})
g.addVertex("Movie", {"name": "The Godfather"})
g.addVertex("Movie", {"name": "The Godfather Part II"})
g.addVertex("Movie", {"name": "The Godfather Coda The Death of Michael Corleone"})

g.addEdge("ActedIn", "1:Al Pacino", "2:The Godfather", {})
g.addEdge("ActedIn", "1:Al Pacino", "2:The Godfather Part II", {})
g.addEdge(
    "ActedIn", "1:Al Pacino", "2:The Godfather Coda The Death of Michael Corleone", {}
)
g.addEdge("ActedIn", "1:Robert De Niro", "2:The Godfather Part II", {})

"""
## Creating `HugeGraphQAChain`

We can now create the `HugeGraph` and `HugeGraphQAChain`. To create the `HugeGraph` we simply need to pass the database object to the `HugeGraph` constructor.
"""
logger.info("## Creating `HugeGraphQAChain`")


graph = HugeGraph(
    username="admin",
    password="admin",
    address="localhost",
    port=8080,
    graph="hugegraph",
)

"""
## Refresh graph schema information

If the schema of database changes, you can refresh the schema information needed to generate Gremlin statements.
"""
logger.info("## Refresh graph schema information")



logger.debug(graph.get_schema)

"""
## Querying the graph

We can now use the graph Gremlin QA chain to ask question of the graph
"""
logger.info("## Querying the graph")

chain = HugeGraphQAChain.from_llm(ChatOllama(model="llama3.2"), graph=graph, verbose=True)

chain.run("Who played in The Godfather?")

logger.info("\n\n[DONE]", bright=True)
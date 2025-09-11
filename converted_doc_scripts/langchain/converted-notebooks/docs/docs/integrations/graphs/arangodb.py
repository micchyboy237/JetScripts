from adb_cloud_connector import get_temp_credentials
from arango import ArangoClient
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import ArangoGraphQAChain
from langchain_community.graphs import ArangoGraph
import json
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
# ArangoDB

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arangodb/interactive_tutorials/blob/master/notebooks/Langchain.ipynb)

>[ArangoDB](https://github.com/arangodb/arangodb) is a scalable graph database system to drive value from
>connected data, faster. Native graphs, an integrated search engine, and JSON support, via
>a single query language. `ArangoDB` runs on-prem or in the cloud.

This notebook shows how to use LLMs to provide a natural language interface to an [ArangoDB](https://github.com/arangodb/arangodb#readme) database.

## Setting up

You can get a local `ArangoDB` instance running via the [ArangoDB Docker image](https://hub.docker.com/_/arangodb):  

```
docker run -p 8529:8529 -e ARANGO_ROOT_PASSWORD= arangodb/arangodb
```

An alternative is to use the [ArangoDB Cloud Connector package](https://github.com/arangodb/adb-cloud-connector#readme) to get a temporary cloud instance running:
"""
logger.info("# ArangoDB")

# %%capture
# %pip install --upgrade --quiet  python-arango # The ArangoDB Python Driver
# %pip install --upgrade --quiet  adb-cloud-connector # The ArangoDB Cloud Instance provisioner
# %pip install --upgrade --quiet  langchain-ollama
# %pip install --upgrade --quiet  langchain



con = get_temp_credentials()

db = ArangoClient(hosts=con["url"]).db(
    con["dbName"], con["username"], con["password"], verify=True
)

logger.debug(json.dumps(con, indent=2))


graph = ArangoGraph(db)

"""
## Populating database

We will rely on the `Python Driver` to import our [GameOfThrones](https://github.com/arangodb/example-datasets/tree/master/GameOfThrones) data into our database.
"""
logger.info("## Populating database")

if db.has_graph("GameOfThrones"):
    db.delete_graph("GameOfThrones", drop_collections=True)

db.create_graph(
    "GameOfThrones",
    edge_definitions=[
        {
            "edge_collection": "ChildOf",
            "from_vertex_collections": ["Characters"],
            "to_vertex_collections": ["Characters"],
        },
    ],
)

documents = [
    {
        "_key": "NedStark",
        "name": "Ned",
        "surname": "Stark",
        "alive": True,
        "age": 41,
        "gender": "male",
    },
    {
        "_key": "CatelynStark",
        "name": "Catelyn",
        "surname": "Stark",
        "alive": False,
        "age": 40,
        "gender": "female",
    },
    {
        "_key": "AryaStark",
        "name": "Arya",
        "surname": "Stark",
        "alive": True,
        "age": 11,
        "gender": "female",
    },
    {
        "_key": "BranStark",
        "name": "Bran",
        "surname": "Stark",
        "alive": True,
        "age": 10,
        "gender": "male",
    },
]

edges = [
    {"_to": "Characters/NedStark", "_from": "Characters/AryaStark"},
    {"_to": "Characters/NedStark", "_from": "Characters/BranStark"},
    {"_to": "Characters/CatelynStark", "_from": "Characters/AryaStark"},
    {"_to": "Characters/CatelynStark", "_from": "Characters/BranStark"},
]

db.collection("Characters").import_bulk(documents)
db.collection("ChildOf").import_bulk(edges)

"""
## Getting and setting the ArangoDB schema

An initial `ArangoDB Schema` is generated upon instantiating the `ArangoDBGraph` object. Below are the schema's getter & setter methods should you be interested in viewing or modifying the schema:
"""
logger.info("## Getting and setting the ArangoDB schema")


logger.debug(json.dumps(graph.schema, indent=4))

graph.set_schema()


logger.debug(json.dumps(graph.schema, indent=4))

"""
## Querying the ArangoDB database

We can now use the `ArangoDB Graph` QA Chain to inquire about our data
"""
logger.info("## Querying the ArangoDB database")


# os.environ["OPENAI_API_KEY"] = "your-key-here"


chain = ArangoGraphQAChain.from_llm(
    ChatOllama(model="llama3.2"), graph=graph, verbose=True
)

chain.run("Is Ned Stark alive?")

chain.run("How old is Arya Stark?")

chain.run("Are Arya Stark and Ned Stark related?")

chain.run("Does Arya Stark have a dead parent?")

"""
## Chain modifiers

You can alter the values of the following `ArangoDBGraphQAChain` class variables to modify the behaviour of your chain results
"""
logger.info("## Chain modifiers")

chain.top_k = 10

chain.return_aql_query = True

chain.return_aql_result = True

chain.max_aql_generation_attempts = 5

chain.aql_examples = """
RETURN DOCUMENT('Characters/NedStark').alive

FOR e IN ChildOf
    FILTER e._from == "Characters/AryaStark" AND e._to == "Characters/NedStark"
    RETURN e
"""

chain.run("Is Ned Stark alive?")

chain.run("Is Bran Stark the child of Ned Stark?")

logger.info("\n\n[DONE]", bright=True)
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import FalkorDBQAChain
from langchain_community.graphs import FalkorDBGraph
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
# FalkorDB

>[FalkorDB](https://www.falkordb.com/) is a low-latency Graph Database that delivers knowledge to GenAI.


This notebook shows how to use LLMs to provide a natural language interface to `FalkorDB` database.


## Setting up

You can run the `falkordb` Docker container locally:

```bash
docker run -p 6379:6379 -it --rm falkordb/falkordb
```

Once launched, you create a database on the local machine and connect to it.
"""
logger.info("# FalkorDB")


"""
## Create a graph connection and insert the demo data
"""
logger.info("## Create a graph connection and insert the demo data")

graph = FalkorDBGraph(database="movies")

graph.query(
    """
    CREATE
        (al:Person {name: 'Al Pacino', birthDate: '1940-04-25'}),
        (robert:Person {name: 'Robert De Niro', birthDate: '1943-08-17'}),
        (tom:Person {name: 'Tom Cruise', birthDate: '1962-07-3'}),
        (val:Person {name: 'Val Kilmer', birthDate: '1959-12-31'}),
        (anthony:Person {name: 'Anthony Edwards', birthDate: '1962-7-19'}),
        (meg:Person {name: 'Meg Ryan', birthDate: '1961-11-19'}),

        (god1:Movie {title: 'The Godfather'}),
        (god2:Movie {title: 'The Godfather: Part II'}),
        (god3:Movie {title: 'The Godfather Coda: The Death of Michael Corleone'}),
        (top:Movie {title: 'Top Gun'}),

        (al)-[:ACTED_IN]->(god1),
        (al)-[:ACTED_IN]->(god2),
        (al)-[:ACTED_IN]->(god3),
        (robert)-[:ACTED_IN]->(god2),
        (tom)-[:ACTED_IN]->(top),
        (val)-[:ACTED_IN]->(top),
        (anthony)-[:ACTED_IN]->(top),
        (meg)-[:ACTED_IN]->(top)
"""
)

"""
## Creating FalkorDBQAChain
"""
logger.info("## Creating FalkorDBQAChain")

graph.refresh_schema()
logger.debug(graph.schema)


# os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"

chain = FalkorDBQAChain.from_llm(ChatOllama(model="llama3.2"), graph=graph, verbose=True)

"""
## Querying the graph
"""
logger.info("## Querying the graph")

chain.run("Who played in Top Gun?")

chain.run("Who is the oldest actor who played in The Godfather: Part II?")

chain.run("Robert De Niro played in which movies?")

logger.info("\n\n[DONE]", bright=True)
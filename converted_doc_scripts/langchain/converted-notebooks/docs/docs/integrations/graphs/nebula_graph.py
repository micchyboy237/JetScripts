from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import NebulaGraphQAChain
from langchain_community.graphs import NebulaGraph
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
# NebulaGraph

>[NebulaGraph](https://www.nebula-graph.io/) is an open-source, distributed, scalable, lightning-fast
> graph database built for super large-scale graphs with milliseconds of latency. It uses the `nGQL` graph query language.
>
>[nGQL](https://docs.nebula-graph.io/3.0.0/3.ngql-guide/1.nGQL-overview/1.overview/) is a declarative graph query language for `NebulaGraph`. It allows expressive and efficient graph patterns. `nGQL` is designed for both developers and operations professionals. `nGQL` is an SQL-like query language.

This notebook shows how to use LLMs to provide a natural language interface to `NebulaGraph` database.

## Setting up

You can start the `NebulaGraph` cluster as a Docker container by running the following script:

```bash
curl -fsSL nebula-up.siwei.io/install.sh | bash
```

Other options are:
- Install as a [Docker Desktop Extension](https://www.docker.com/blog/distributed-cloud-native-graph-database-nebulagraph-docker-extension/). See [here](https://docs.nebula-graph.io/3.5.0/2.quick-start/1.quick-start-workflow/)
- NebulaGraph Cloud Service. See [here](https://www.nebula-graph.io/cloud)
- Deploy from package, source code, or via Kubernetes. See [here](https://docs.nebula-graph.io/)

Once the cluster is running, we could create the `SPACE` and `SCHEMA` for the database.
"""
logger.info("# NebulaGraph")

# %pip install --upgrade --quiet  ipython-ngql
# %load_ext ngql

# %ngql --address 127.0.0.1 --port 9669 --user root --password nebula
# %ngql CREATE SPACE IF NOT EXISTS langchain(partition_num=1, replica_factor=1, vid_type=fixed_string(128));

# %ngql USE langchain;

"""
Create the schema, for full dataset, refer [here](https://www.siwei.io/en/nebulagraph-etl-dbt/).
"""
logger.info("Create the schema, for full dataset, refer [here](https://www.siwei.io/en/nebulagraph-etl-dbt/).")

# %%ngql
CREATE TAG IF NOT EXISTS movie(name string);
CREATE TAG IF NOT EXISTS person(name string, birthdate string);
CREATE EDGE IF NOT EXISTS acted_in();
CREATE TAG INDEX IF NOT EXISTS person_index ON person(name(128));
CREATE TAG INDEX IF NOT EXISTS movie_index ON movie(name(128));

"""
Wait for schema creation to complete, then we can insert some data.
"""
logger.info("Wait for schema creation to complete, then we can insert some data.")

# %%ngql
INSERT VERTEX person(name, birthdate) VALUES "Al Pacino":("Al Pacino", "1940-04-25");
INSERT VERTEX movie(name) VALUES "The Godfather II":("The Godfather II");
INSERT VERTEX movie(name) VALUES "The Godfather Coda: The Death of Michael Corleone":("The Godfather Coda: The Death of Michael Corleone");
INSERT EDGE acted_in() VALUES "Al Pacino"->"The Godfather II":();
INSERT EDGE acted_in() VALUES "Al Pacino"->"The Godfather Coda: The Death of Michael Corleone":();


graph = NebulaGraph(
    space="langchain",
    username="root",
    password="nebula",
    address="127.0.0.1",
    port=9669,
    session_pool_size=30,
)

"""
## Refresh graph schema information

If the schema of database changes, you can refresh the schema information needed to generate nGQL statements.
"""
logger.info("## Refresh graph schema information")



logger.debug(graph.get_schema)

"""
## Querying the graph

We can now use the graph cypher QA chain to ask question of the graph
"""
logger.info("## Querying the graph")

chain = NebulaGraphQAChain.from_llm(
    ChatOllama(model="llama3.2"), graph=graph, verbose=True
)

chain.run("Who played in The Godfather II?")

logger.info("\n\n[DONE]", bright=True)
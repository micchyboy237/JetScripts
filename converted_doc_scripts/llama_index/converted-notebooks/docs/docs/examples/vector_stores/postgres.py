from datetime import datetime
from jet.logger import logger
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
MetadataFilter,
MetadataFilters,
)
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import (
Table,
MetaData,
Column,
String,
Integer,
create_engine,
insert,
)
from sqlalchemy import Select
from sqlalchemy import make_url
from typing import Any
import csv
import os
import psycopg2
import re
import shutil
import textwrap


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/postgres.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Postgres Vector Store
In this notebook we are going to show how to use [Postgresql](https://www.postgresql.org) and  [pgvector](https://github.com/pgvector/pgvector)  to perform vector searches in LlamaIndex

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Postgres Vector Store")

# %pip install llama-index-vector-stores-postgres

# !pip install llama-index

"""
Running the following cell will install Postgres with PGVector in Colab.
"""
logger.info("Running the following cell will install Postgres with PGVector in Colab.")

# !sudo apt update
# !echo | sudo apt install -y postgresql-common
# !echo | sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
# !echo | sudo apt install postgresql-15-pgvector
# !sudo service postgresql start
# !sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'password';"
# !sudo -u postgres psql -c "CREATE DATABASE vector_db;"


"""
### Setup Ollama
The first step is to configure the ollama key. It will be used to created embeddings for the documents loaded into the index
"""
logger.info("### Setup Ollama")


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Loading documents
Load the documents stored in the `data/paul_graham/` using the SimpleDirectoryReader
"""
logger.info("### Loading documents")

documents = SimpleDirectoryReader("./data/paul_graham").load_data()
logger.debug("Document ID:", documents[0].doc_id)

"""
### Create the Database
Using an existing postgres running at localhost, create the database we'll be using.
"""
logger.info("### Create the Database")


connection_string = "postgresql://postgres:password@localhost:5432"
db_name = "vector_db"
conn = psycopg2.connect(connection_string)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")

"""
### Create the index
Here we create an index backed by Postgres using the documents loaded previously. PGVectorStore takes a few arguments. The example below constructs a PGVectorStore with a HNSW index with m = 16, ef_construction = 64, and ef_search = 40, with the `vector_cosine_ops` method.
"""
logger.info("### Create the index")


url = make_url(connection_string)
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="paul_graham_essay",
    embed_dim=1536,  # ollama embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)
query_engine = index.as_query_engine()

"""
### Query the index
We can now ask questions using our index.
"""
logger.info("### Query the index")

response = query_engine.query("What did the author do?")

logger.debug(textwrap.fill(str(response), 100))

response = query_engine.query("What happened in the mid 1980s?")

logger.debug(textwrap.fill(str(response), 100))

"""
### Querying existing index
"""
logger.info("### Querying existing index")

vector_store = PGVectorStore.from_params(
    database="vector_db",
    host="localhost",
    password="password",
    port=5432,
    user="postgres",
    table_name="paul_graham_essay",
    embed_dim=1536,  # ollama embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine()

response = query_engine.query("What did the author do?")

logger.debug(textwrap.fill(str(response), 100))

"""
### Hybrid Search

To enable hybrid search, you need to:
1. pass in `hybrid_search=True` when constructing the `PGVectorStore` (and optionally configure `text_search_config` with the desired language)
2. pass in `vector_store_query_mode="hybrid"` when constructing the query engine (this config is passed to the retriever under the hood). You can also optionally set the `sparse_top_k` to configure how many results we should obtain from sparse text search (default is using the same value as `similarity_top_k`).
"""
logger.info("### Hybrid Search")


url = make_url(connection_string)
hybrid_vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="paul_graham_essay_hybrid_search",
    embed_dim=1536,  # ollama embedding dimension
    hybrid_search=True,
    text_search_config="english",
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

storage_context = StorageContext.from_defaults(
    vector_store=hybrid_vector_store
)
hybrid_index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

hybrid_query_engine = hybrid_index.as_query_engine(
    vector_store_query_mode="hybrid", sparse_top_k=2
)
hybrid_response = hybrid_query_engine.query(
    "Who does Paul Graham think of with the word schtick"
)

logger.debug(hybrid_response)

"""
#### Improving hybrid search with QueryFusionRetriever

Since the scores for text search and vector search are calculated differently, the nodes that were found only by text search will have a much lower score.

You can often improve hybrid search performance by using `QueryFusionRetriever`, which makes better use of the mutual information to rank the nodes.
"""
logger.info("#### Improving hybrid search with QueryFusionRetriever")


vector_retriever = hybrid_index.as_retriever(
    vector_store_query_mode="default",
    similarity_top_k=5,
)
text_retriever = hybrid_index.as_retriever(
    vector_store_query_mode="sparse",
    similarity_top_k=5,  # interchangeable with sparse_top_k in this context
)
retriever = QueryFusionRetriever(
    [vector_retriever, text_retriever],
    similarity_top_k=5,
    num_queries=1,  # set this to 1 to disable query generation
    mode="relative_score",
    use_async=False,
)

response_synthesizer = CompactAndRefine()
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

response = query_engine.query(
    "Who does Paul Graham think of with the word schtick, and why?"
)
logger.debug(response)

"""
### Metadata filters

PGVectorStore supports storing metadata in nodes, and filtering based on that metadata during the retrieval step.

#### Download git commits dataset
"""
logger.info("### Metadata filters")

# !mkdir -p 'data/git_commits/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/csv/commit_history.csv' -O 'data/git_commits/commit_history.csv'


with open("data/git_commits/commit_history.csv", "r") as f:
    commits = list(csv.DictReader(f))

logger.debug(commits[0])
logger.debug(len(commits))

"""
#### Add nodes with custom metadata
"""
logger.info("#### Add nodes with custom metadata")


nodes = []
dates = set()
authors = set()
for commit in commits[:100]:
    author_email = commit["author"].split("<")[1][:-1]
    commit_date = datetime.strptime(
        commit["date"], "%a %b %d %H:%M:%S %Y %z"
    ).strftime("%Y-%m-%d")
    commit_text = commit["change summary"]
    if commit["change details"]:
        commit_text += "\n\n" + commit["change details"]
    fixes = re.findall(r"#(\d+)", commit_text, re.IGNORECASE)
    nodes.append(
        TextNode(
            text=commit_text,
            metadata={
                "commit_date": commit_date,
                "author": author_email,
                "fixes": fixes,
            },
        )
    )
    dates.add(commit_date)
    authors.add(author_email)

logger.debug(nodes[0])
logger.debug(min(dates), "to", max(dates))
logger.debug(authors)

vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="metadata_filter_demo3",
    embed_dim=1536,  # ollama embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
index.insert_nodes(nodes)

logger.debug(index.as_query_engine().query("How did Lakshmi fix the segfault?"))

"""
#### Apply metadata filters

Now we can filter by commit author or by date when retrieving nodes.
"""
logger.info("#### Apply metadata filters")


filters = MetadataFilters(
    filters=[
        MetadataFilter(key="author", value="mats@timescale.com"),
        MetadataFilter(key="author", value="sven@timescale.com"),
    ],
    condition="or",
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("What is this software project about?")

for node in retrieved_nodes:
    logger.debug(node.node.metadata)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="commit_date", value="2023-08-15", operator=">="),
        MetadataFilter(key="commit_date", value="2023-08-25", operator="<="),
    ],
    condition="and",
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("What is this software project about?")

for node in retrieved_nodes:
    logger.debug(node.node.metadata)

"""
#### Apply nested filters

In the above examples, we combined multiple filters using AND or OR. We can also combine multiple sets of filters.

e.g. in SQL:
```sql
WHERE (commit_date >= '2023-08-01' AND commit_date <= '2023-08-15') AND (author = 'mats@timescale.com' OR author = 'sven@timescale.com')
```
"""
logger.info("#### Apply nested filters")

filters = MetadataFilters(
    filters=[
        MetadataFilters(
            filters=[
                MetadataFilter(
                    key="commit_date", value="2023-08-01", operator=">="
                ),
                MetadataFilter(
                    key="commit_date", value="2023-08-15", operator="<="
                ),
            ],
            condition="and",
        ),
        MetadataFilters(
            filters=[
                MetadataFilter(key="author", value="mats@timescale.com"),
                MetadataFilter(key="author", value="sven@timescale.com"),
            ],
            condition="or",
        ),
    ],
    condition="and",
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("What is this software project about?")

for node in retrieved_nodes:
    logger.debug(node.node.metadata)

"""
The above can be simplified by using the IN operator. `PGVectorStore` supports `in`, `nin`, and `contains` for comparing an element with a list.
"""
logger.info("The above can be simplified by using the IN operator. `PGVectorStore` supports `in`, `nin`, and `contains` for comparing an element with a list.")

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="commit_date", value="2023-08-01", operator=">="),
        MetadataFilter(key="commit_date", value="2023-08-15", operator="<="),
        MetadataFilter(
            key="author",
            value=["mats@timescale.com", "sven@timescale.com"],
            operator="in",
        ),
    ],
    condition="and",
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("What is this software project about?")

for node in retrieved_nodes:
    logger.debug(node.node.metadata)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="commit_date", value="2023-08-01", operator=">="),
        MetadataFilter(key="commit_date", value="2023-08-15", operator="<="),
        MetadataFilter(
            key="author",
            value=["mats@timescale.com", "sven@timescale.com"],
            operator="nin",
        ),
    ],
    condition="and",
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("What is this software project about?")

for node in retrieved_nodes:
    logger.debug(node.node.metadata)

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="fixes", value="5680", operator="contains"),
    ]
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("How did these commits fix the issue?")
for node in retrieved_nodes:
    logger.debug(node.node.metadata)

"""
### Customize queries

It is possible to build more complex queries such as joining other tables. This is done by setting the `customize_query_fn` argument with your function. First, lets create a user table and populate it.
"""
logger.info("### Customize queries")


engine = create_engine(url=connection_string + "/" + db_name)

metadata = MetaData()

user_table = Table(
    "user",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False),
    Column("email", String, nullable=False),
)

user_table.drop(engine, checkfirst=True)
user_table.create(engine)

with engine.begin() as conn:
    stmt = insert(user_table)
    conn.execute(
        stmt, [{"name": "Konstantina", "email": "konstantina@timescale.com"}]
    )

"""
Then, we can create a query customization function and instantiate `PGVectorStore` with `customize_query_fn`.
"""
logger.info("Then, we can create a query customization function and instantiate `PGVectorStore` with `customize_query_fn`.")



def customize_query(query: Select, table_class: Any, **kwargs: Any) -> Select:
    return query.add_columns(user_table.c.name).join(
        user_table,
        user_table.c.email == table_class.metadata_["author"].astext,
    )


vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="metadata_filter_demo3",
    embed_dim=1536,  # ollama embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
    customize_query_fn=customize_query,
)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

"""
We can then query the vector store and retrieve any additional field added to the select statement in a dictionary named `custom_fields` in the node metadata.
"""
logger.info("We can then query the vector store and retrieve any additional field added to the select statement in a dictionary named `custom_fields` in the node metadata.")

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="fixes", value="5680", operator="contains"),
    ]
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=filters,
)

retrieved_nodes = retriever.retrieve("How did these commits fix the issue?")
for node in retrieved_nodes:
    logger.debug(node.node.metadata)

"""
### PgVector Query Options

#### IVFFlat Probes

Specify the number of [IVFFlat probes](https://github.com/pgvector/pgvector?tab=readme-ov-file#query-options) (1 by default)

When retrieving from the index, you can specify an appropriate number of IVFFlat probes (higher is better for recall, lower is better for speed)
"""
logger.info("### PgVector Query Options")

retriever = index.as_retriever(
    vector_store_query_mode="hybrid",
    similarity_top_k=5,
    vector_store_kwargs={"ivfflat_probes": 10},
)

"""
#### HNSW EF Search

Specify the size of the dynamic [candidate list](https://github.com/pgvector/pgvector?tab=readme-ov-file#query-options-1) for search (40 by default)
"""
logger.info("#### HNSW EF Search")

retriever = index.as_retriever(
    vector_store_query_mode="hybrid",
    similarity_top_k=5,
    vector_store_kwargs={"hnsw_ef_search": 300},
)

logger.info("\n\n[DONE]", bright=True)
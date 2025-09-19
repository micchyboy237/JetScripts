from jet.models.config import MODELS_CACHE_DIR
from IPython.display import Markdown, display
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import SQLDatabase
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings.openai import HuggingFaceEmbedding
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import NLSQLRetriever
from llama_index.core.schema import TextNode
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)
from sqlalchemy import insert
from sqlalchemy import text
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/index_structs/struct_indices/SQLIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Text-to-SQL Guide (Query Engine + Retriever)

This is a basic guide to LlamaIndex's Text-to-SQL capabilities. 
1. We first show how to perform text-to-SQL over a toy dataset: this will do "retrieval" (sql query over db) and "synthesis".
2. We then show how to buid a TableIndex over the schema to dynamically retrieve relevant tables during query-time.
3. Next, we show how to use query-time rows and columns retrievers to enhance Text-to-SQL context.
4. We finally show you how to define a text-to-SQL retriever on its own.

**NOTE:** Any Text-to-SQL application should be aware that executing 
arbitrary SQL queries can be a security risk. It is recommended to
take precautions as needed, such as using restricted roles, read-only
databases, sandboxing, etc.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Text-to-SQL Guide (Query Engine + Retriever)")

# %pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-huggingface


# os.environ["OPENAI_API_KEY"] = "sk-.."


"""
### Create Database Schema

We use `sqlalchemy`, a popular SQL database toolkit, to create an empty `city_stats` Table
"""
logger.info("### Create Database Schema")


engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
metadata_obj.create_all(engine)

"""
### Define SQL Database

We first define our `SQLDatabase` abstraction (a light wrapper around SQLAlchemy).
"""
logger.info("### Define SQL Database")


llm = OllamaFunctionCalling(temperature=0.1, model="llama3.2")

sql_database = SQLDatabase(engine, include_tables=["city_stats"])

"""
We add some testing data to our SQL database.
"""
logger.info("We add some testing data to our SQL database.")


sql_database = SQLDatabase(engine, include_tables=["city_stats"])

rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {
        "city_name": "Chicago",
        "population": 2679000,
        "country": "United States",
    },
    {
        "city_name": "New York",
        "population": 8258000,
        "country": "United States",
    },
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
    {"city_name": "Busan", "population": 3334000, "country": "South Korea"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

stmt = select(
    city_stats_table.c.city_name,
    city_stats_table.c.population,
    city_stats_table.c.country,
).select_from(city_stats_table)

with engine.connect() as connection:
    results = connection.execute(stmt).fetchall()
    logger.debug(results)

"""
### Query Index

We first show how we can execute a raw SQL query, which directly executes over the table.
"""
logger.info("### Query Index")


with engine.connect() as con:
    rows = con.execute(text("SELECT city_name from city_stats"))
    for row in rows:
        logger.debug(row)

"""
## Part 1: Text-to-SQL Query Engine
Once we have constructed our SQL database, we can use the NLSQLTableQueryEngine to
construct natural language queries that are synthesized into SQL queries.

Note that we need to specify the tables we want to use with this query engine.
If we don't the query engine will pull all the schema context, which could
overflow the context window of the LLM.
"""
logger.info("## Part 1: Text-to-SQL Query Engine")


query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database, tables=["city_stats"], llm=llm
)
query_str = "Which city has the highest population?"
response = query_engine.query(query_str)

display(Markdown(f"<b>{response}</b>"))

"""
This query engine should be used in any case where you can specify the tables you want
to query over beforehand, or the total size of all the table schema plus the rest of
the prompt fits your context window.

## Part 2: Query-Time Retrieval of Tables for Text-to-SQL
If we don't know ahead of time which table we would like to use, and the total size of
the table schema overflows your context window size, we should store the table schema 
in an index so that during query time we can retrieve the right schema.

The way we can do this is using the SQLTableNodeMapping object, which takes in a 
SQLDatabase and produces a Node object for each SQLTableSchema object passed 
into the ObjectIndex constructor.
"""
logger.info("## Part 2: Query-Time Retrieval of Tables for Text-to-SQL")


table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    (SQLTableSchema(table_name="city_stats"))
]  # add a SQLTableSchema for each table

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
    embed_model=HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
)
query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=1)
)

"""
Now we can take our SQLTableRetrieverQueryEngine and query it for our response.
"""
logger.info(
    "Now we can take our SQLTableRetrieverQueryEngine and query it for our response.")

response = query_engine.query("Which city has the highest population?")
display(Markdown(f"<b>{response}</b>"))

response.metadata["result"]

"""
You can also add additional context information for each table schema you define.
"""
logger.info(
    "You can also add additional context information for each table schema you define.")

city_stats_text = (
    "This table gives information regarding the population and country of a"
    " given city.\nThe user will query with codewords, where 'foo' corresponds"
    " to population and 'bar'corresponds to city."
)

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    (SQLTableSchema(table_name="city_stats", context_str=city_stats_text))
]

"""
## Part 3: Query-Time Retrieval of Rows and Columns for Text-to-SQL

One challenge arises when asking a question like: "How many cities are in the US?" In such cases, the generated query might only look for cities where the country is listed as "US," potentially missing entries labeled "United States." To address this issue, you can apply query-time row retrieval, query-time column retrieval, or a combination of both.

### Query-Time Rows Retrieval

In query-time rows retrieval, we embed the rows of each table, resulting in one index per table.
"""
logger.info("## Part 3: Query-Time Retrieval of Rows and Columns for Text-to-SQL")


with engine.connect() as connection:
    results = connection.execute(stmt).fetchall()

city_nodes = [TextNode(text=str(t)) for t in results]

city_rows_index = VectorStoreIndex(
    city_nodes, embed_model=HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
)
city_rows_retriever = city_rows_index.as_retriever(similarity_top_k=1)

city_rows_retriever.retrieve("US")

"""
Then, the rows retriever of each table can be provided to the SQLTableRetrieverQueryEngine.
"""
logger.info(
    "Then, the rows retriever of each table can be provided to the SQLTableRetrieverQueryEngine.")

rows_retrievers = {
    "city_stats": city_rows_retriever,
}
query_engine = SQLTableRetrieverQueryEngine(
    sql_database,
    obj_index.as_retriever(similarity_top_k=1),
    rows_retrievers=rows_retrievers,
)

"""
During querying, the row retrievers are used to identify the rows most semantically similar to the input query. These retrieved rows are then incorporated as context to enhance the performance of the text-to-SQL generation.
"""
logger.info("During querying, the row retrievers are used to identify the rows most semantically similar to the input query. These retrieved rows are then incorporated as context to enhance the performance of the text-to-SQL generation.")

response = query_engine.query("How many cities are in the US?")

display(Markdown(f"<b>{response}</b>"))

"""
### Query-Time Columns Retrieval
While query-time row retrieval enhances text-to-SQL generation, it embeds each row individually, even when many rows share repeated values—such as those in categorical. This can lead to token inefficiency and unnecessary overhead. Moreover, in tables with a large number of columns, the retriever may surface only a subset of relevant values, potentially omitting others that are important for accurate query generation.

To address this issue, query-time column retrieval can be used. This approach indexes each distinct value within selected columns, creating a separate index for each column in the table.
"""
logger.info("### Query-Time Columns Retrieval")

city_cols_retrievers = {}

for column_name in ["city_name", "country"]:
    stmt = select(city_stats_table.c[column_name]).distinct()
    with engine.connect() as connection:
        values = connection.execute(stmt).fetchall()
    nodes = [TextNode(text=t[0]) for t in values]

    column_index = VectorStoreIndex(
        nodes, embed_model=HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
    )
    column_retriever = column_index.as_retriever(similarity_top_k=1)

    city_cols_retrievers[column_name] = column_retriever

"""
Then, columns retrievers of each table can be provided to the SQLTableRetrieverQueryEngine.
"""
logger.info(
    "Then, columns retrievers of each table can be provided to the SQLTableRetrieverQueryEngine.")

cols_retrievers = {
    "city_stats": city_cols_retrievers,
}
query_engine = SQLTableRetrieverQueryEngine(
    sql_database,
    obj_index.as_retriever(similarity_top_k=1),
    rows_retrievers=rows_retrievers,
    cols_retrievers=cols_retrievers,
    llm=llm,
)

"""
During querying, the columns retrievers are used to identify the values of columns that are the most semantically similar to the input query. These retrieved values are then incorporated as context to enhance the performance of the text-to-SQL generation.
"""
logger.info("During querying, the columns retrievers are used to identify the values of columns that are the most semantically similar to the input query. These retrieved values are then incorporated as context to enhance the performance of the text-to-SQL generation.")

response = query_engine.query("How many cities are in the US?")

display(Markdown(f"<b>{response}</b>"))

"""
## Part 4: Text-to-SQL Retriever

So far our text-to-SQL capability is packaged in a query engine and consists of both retrieval and synthesis.

You can use the SQL retriever on its own. We show you some different parameters you can try, and also show how to plug it into our `RetrieverQueryEngine` to get roughly the same results.
"""
logger.info("## Part 4: Text-to-SQL Retriever")


nl_sql_retriever = NLSQLRetriever(
    sql_database, tables=["city_stats"], llm=llm, return_raw=True
)

results = nl_sql_retriever.retrieve(
    "Return the top 5 cities (along with their populations) with the highest population."
)


for n in results:
    display_source_node(n)

nl_sql_retriever = NLSQLRetriever(
    sql_database, tables=["city_stats"], return_raw=False
)

results = nl_sql_retriever.retrieve(
    "Return the top 5 cities (along with their populations) with the highest population."
)

for n in results:
    display_source_node(n, show_source_metadata=True)

"""
### Plug into our `RetrieverQueryEngine`

We compose our SQL Retriever with our standard `RetrieverQueryEngine` to synthesize a response. The result is roughly similar to our packaged `Text-to-SQL` query engines.
"""
logger.info("### Plug into our `RetrieverQueryEngine`")


query_engine = RetrieverQueryEngine.from_args(nl_sql_retriever, llm=llm)

response = query_engine.query(
    "Return the top 5 cities (along with their populations) with the highest population."
)

logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)

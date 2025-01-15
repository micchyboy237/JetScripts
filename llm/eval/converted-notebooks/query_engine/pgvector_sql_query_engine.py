"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/pgvector_sql_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# [Beta] Text-to-SQL with PGVector

This notebook demo shows how to perform text-to-SQL with pgvector. This allows us to jointly do both semantic search and structured querying, *all* within SQL!

This hypothetically enables more expressive queries than semantic search + metadata filters.

**NOTE**: This is a beta feature, interfaces might change. But in the meantime hope you find it useful! 

**NOTE:** Any Text-to-SQL application should be aware that executing 
arbitrary SQL queries can be a security risk. It is recommended to
take precautions as needed, such as using restricted roles, read-only
databases, sandboxing, etc.
"""

"""
## Setup Data
"""

"""
### Load Documents

Load in the Lyft 2021 10k document.
"""

# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-readers-file
# %pip install llama-index-llms-ollama


from llama_index.core import Settings
from llama_index.core.query_engine import PGVectorSQLQueryEngine
from jet.llm.ollama import Ollama
from llama_index.core import SQLDatabase
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy.orm import declarative_base, mapped_column
from sqlalchemy import insert, create_engine, String, text, Integer
from pgvector.sqlalchemy import Vector
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
reader = PDFReader()

"""
Download Data
"""

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

docs = reader.load_data("./data/10k/lyft_2021.pdf")


node_parser = SentenceSplitter()
nodes = node_parser.get_nodes_from_documents(docs)

print(nodes[8].get_content(metadata_mode="all"))

"""
### Insert data into Postgres + PGVector

Make sure you have all the necessary dependencies installed!
"""

# !pip install psycopg2-binary pgvector asyncpg "sqlalchemy[asyncio]" greenlet


"""
#### Establish Connection
"""

engine = create_engine("postgresql+psycopg2://localhost/postgres")
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

"""
#### Define Table Schema 

Define as Python class. Note we store the page_label, embedding, and text.
"""

Base = declarative_base()


class SECTextChunk(Base):
    __tablename__ = "sec_text_chunk"

    id = mapped_column(Integer, primary_key=True)
    page_label = mapped_column(Integer)
    file_name = mapped_column(String)
    text = mapped_column(String)
    embedding = mapped_column(Vector(384))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

"""
#### Generate embedding for each Node with a sentence_transformers model
"""


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

for node in nodes:
    text_embedding = embed_model.get_text_embedding(node.get_content())
    node.embedding = text_embedding

"""
#### Insert into Database
"""

for node in nodes:
    row_dict = {
        "text": node.get_content(),
        "embedding": node.embedding,
        **node.metadata,
    }
    stmt = insert(SECTextChunk).values(**row_dict)
    with engine.connect() as connection:
        cursor = connection.execute(stmt)
        connection.commit()

"""
## Define PGVectorSQLQueryEngine

Now that we've loaded the data into the database, we're ready to setup our query engine.
"""

"""
### Define Prompt

We create a modified version of our default text-to-SQL prompt to inject awareness of the pgvector syntax.
We also prompt it with some few-shot examples of how to use the syntax (<-->). 

**NOTE**: This is included by default in the `PGVectorSQLQueryEngine`, we included it here mostly for visibility!
"""


text_to_sql_tmpl = """\
Given an input question, first create a syntactically correct {dialect} \
query to run, then look at the results of the query and return the answer. \
You can order the results by a relevant column to return the most \
interesting examples in the database.

Pay attention to use only the column names that you can see in the schema \
description. Be careful to not query for columns that do not exist. \
Pay attention to which column is in which table. Also, qualify column names \
with the table name when needed. 

IMPORTANT NOTE: you can use specialized pgvector syntax (`<->`) to do nearest \
neighbors/semantic search to a given vector from an embeddings column in the table. \
The embeddings value for a given row typically represents the semantic meaning of that row. \
The vector represents an embedding representation \
of the question, given below. Do NOT fill in the vector values directly, but rather specify a \
`[query_vector]` placeholder. For instance, some select statement examples below \
(the name of the embeddings column is `embedding`):
SELECT * FROM items ORDER BY embedding <-> '[query_vector]' LIMIT 5;
SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 5;
SELECT * FROM items WHERE embedding <-> '[query_vector]' < 5;

You are required to use the following format, \
each taking one line:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use tables listed below.
{schema}


Question: {query_str}
SQLQuery: \
"""
text_to_sql_prompt = PromptTemplate(text_to_sql_tmpl)

"""
### Setup LLM, Embedding Model, and Misc.

Besides LLM and embedding model, note we also add annotations on the table itself. This better helps the LLM 
understand the column schema (e.g. by telling it what the embedding column represents) to better do 
either tabular querying or semantic search.
"""


sql_database = SQLDatabase(engine, include_tables=["sec_text_chunk"])

Settings.llm = Ollama(
    model="llama3.1", request_timeout=300.0, context_window=4096)
Settings.embed_model = embed_model


table_desc = """\
This table represents text chunks from an SEC filing. Each row contains the following columns:

id: id of row
page_label: page number 
file_name: top-level file name
text: all text chunk is here
embedding: the embeddings representing the text chunk

For most queries you should perform semantic search against the `embedding` column values, since \
that encodes the meaning of the text.

"""

context_query_kwargs = {"sec_text_chunk": table_desc}

"""
### Define Query Engine
"""

query_engine = PGVectorSQLQueryEngine(
    sql_database=sql_database,
    text_to_sql_prompt=text_to_sql_prompt,
    context_query_kwargs=context_query_kwargs,
)

"""
## Run Some Queries

Now we're ready to run some queries
"""

response = query_engine.query(
    "Can you tell me about the risk factors described in page 6?",
)

print(str(response))

print(response.metadata["sql_query"])

response = query_engine.query(
    "Tell me more about Lyft's real estate operating leases",
)

print(str(response))

print(response.metadata["sql_query"][:300])

print(response.metadata["result"])

response = query_engine.query(
    "Tell me about the max page number in this table",
)

print(str(response))

print(response.metadata["sql_query"][:300])

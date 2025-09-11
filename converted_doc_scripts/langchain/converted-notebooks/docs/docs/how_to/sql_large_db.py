from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers.ollama_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from pydantic import BaseModel, Field
from typing import List
import ChatModelTabs from "@theme/ChatModelTabs";
import ast
import os
import re
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
# How to deal with large databases when doing SQL question-answering

In order to write valid queries against a database, we need to feed the model the table names, table schemas, and feature values for it to query over. When there are many tables, columns, and/or high-cardinality columns, it becomes impossible for us to dump the full information about our database in every prompt. Instead, we must find ways to dynamically insert into the prompt only the most relevant information.

In this guide we demonstrate methods for identifying such relevant information, and feeding this into a query-generation step. We will cover:

1. Identifying a relevant subset of tables;
2. Identifying a relevant subset of column values.


## Setup

First, get required packages and set environment variables:
"""
logger.info("# How to deal with large databases when doing SQL question-answering")

# %pip install --upgrade --quiet  langchain langchain-community langchain-ollama


"""
The below example will use a SQLite connection with Chinook database. Follow [these installation steps](https://database.guide/2-sample-databases-sqlite/) to create `Chinook.db` in the same directory as this notebook:

* Save [this file](https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql) as `Chinook_Sqlite.sql`
* Run `sqlite3 Chinook.db`
* Run `.read Chinook_Sqlite.sql`
* Test `SELECT * FROM Artist LIMIT 10;`

Now, `Chinook.db` is in our directory and we can interface with it using the SQLAlchemy-driven [SQLDatabase](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.sql_database.SQLDatabase.html) class:
"""
logger.info("The below example will use a SQLite connection with Chinook database. Follow [these installation steps](https://database.guide/2-sample-databases-sqlite/) to create `Chinook.db` in the same directory as this notebook:")


db = SQLDatabase.from_uri("sqlite:///Chinook.db")
logger.debug(db.dialect)
logger.debug(db.get_usable_table_names())
logger.debug(db.run("SELECT * FROM Artist LIMIT 10;"))

"""
## Many tables

One of the main pieces of information we need to include in our prompt is the schemas of the relevant tables. When we have very many tables, we can't fit all of the schemas in a single prompt. What we can do in such cases is first extract the names of the tables related to the user input, and then include only their schemas.

One easy and reliable way to do this is using [tool-calling](/docs/how_to/tool_calling). Below, we show how we can use this feature to obtain output conforming to a desired format (in this case, a list of table names). We use the chat model's `.bind_tools` method to bind a tool in Pydantic format, and feed this into an output parser to reconstruct the object from the model's response.


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Many tables")


llm = ChatOllama(model="llama3.2")



class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")


table_names = "\n".join(db.get_usable_table_names())
system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_names}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{input}"),
    ]
)
llm_with_tools = llm.bind_tools([Table])
output_parser = PydanticToolsParser(tools=[Table])

table_chain = prompt | llm_with_tools | output_parser

table_chain.invoke({"input": "What are all the genres of Alanis Morissette songs"})

"""
This works pretty well! Except, as we'll see below, we actually need a few other tables as well. This would be pretty difficult for the model to know based just on the user question. In this case, we might think to simplify our model's job by grouping the tables together. We'll just ask the model to choose between categories "Music" and "Business", and then take care of selecting all the relevant tables from there:
"""
logger.info("This works pretty well! Except, as we'll see below, we actually need a few other tables as well. This would be pretty difficult for the model to know based just on the user question. In this case, we might think to simplify our model's job by grouping the tables together. We'll just ask the model to choose between categories "Music" and "Business", and then take care of selecting all the relevant tables from there:")

system = """Return the names of any SQL tables that are relevant to the user question.
The tables are:

Music
Business
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{input}"),
    ]
)

category_chain = prompt | llm_with_tools | output_parser
category_chain.invoke({"input": "What are all the genres of Alanis Morissette songs"})



def get_tables(categories: List[Table]) -> List[str]:
    tables = []
    for category in categories:
        if category.name == "Music":
            tables.extend(
                [
                    "Album",
                    "Artist",
                    "Genre",
                    "MediaType",
                    "Playlist",
                    "PlaylistTrack",
                    "Track",
                ]
            )
        elif category.name == "Business":
            tables.extend(["Customer", "Employee", "Invoice", "InvoiceLine"])
    return tables


table_chain = category_chain | get_tables
table_chain.invoke({"input": "What are all the genres of Alanis Morissette songs"})

"""
Now that we've got a chain that can output the relevant tables for any query we can combine this with our [create_sql_query_chain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.sql_database.query.create_sql_query_chain.html), which can accept a list of `table_names_to_use` to determine which table schemas are included in the prompt:
"""
logger.info("Now that we've got a chain that can output the relevant tables for any query we can combine this with our [create_sql_query_chain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.sql_database.query.create_sql_query_chain.html), which can accept a list of `table_names_to_use` to determine which table schemas are included in the prompt:")



query_chain = create_sql_query_chain(llm, db)
table_chain = {"input": itemgetter("question")} | table_chain
full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain

query = full_chain.invoke(
    {"question": "What are all the genres of Alanis Morissette songs"}
)
logger.debug(query)

db.run(query)

"""
We can see the LangSmith trace for this run [here](https://smith.langchain.com/public/4fbad408-3554-4f33-ab47-1e510a1b52a3/r).

We've seen how to dynamically include a subset of table schemas in a prompt within a chain. Another possible approach to this problem is to let an Agent decide for itself when to look up tables by giving it a Tool to do so. You can see an example of this in the [SQL: Agents](/docs/tutorials/sql_qa/#agents) guide.

## High-cardinality columns

In order to filter columns that contain proper nouns such as addresses, song names or artists, we first need to double-check the spelling in order to filter the data correctly. 

One naive strategy it to create a vector store with all the distinct proper nouns that exist in the database. We can then query that vector store each user input and inject the most relevant proper nouns into the prompt.

First we need the unique values for each entity we want, for which we define a function that parses the result into a list of elements:
"""
logger.info("## High-cardinality columns")



def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return res


proper_nouns = query_as_list(db, "SELECT Name FROM Artist")
proper_nouns += query_as_list(db, "SELECT Title FROM Album")
proper_nouns += query_as_list(db, "SELECT Name FROM Genre")
len(proper_nouns)
proper_nouns[:5]

"""
Now we can embed and store all of our values in a vector database:
"""
logger.info("Now we can embed and store all of our values in a vector database:")


vector_db = FAISS.from_texts(proper_nouns, OllamaEmbeddings(model="mxbai-embed-large"))
retriever = vector_db.as_retriever(search_kwargs={"k": 15})

"""
And put together a query construction chain that first retrieves values from the database and inserts them into the prompt:
"""
logger.info("And put together a query construction chain that first retrieves values from the database and inserts them into the prompt:")



system = """You are a SQLite expert. Given an input question, create a syntactically
correct SQLite query to run. Unless otherwise specificed, do not return more than
{top_k} rows.

Only return the SQL query with no markup or explanation.

Here is the relevant table info: {table_info}

Here is a non-exhaustive list of possible feature values. If filtering on a feature
value make sure to check its spelling against this list first:

{proper_nouns}
"""

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

query_chain = create_sql_query_chain(llm, db, prompt=prompt)
retriever_chain = (
    itemgetter("question")
    | retriever
    | (lambda docs: "\n".join(doc.page_content for doc in docs))
)
chain = RunnablePassthrough.assign(proper_nouns=retriever_chain) | query_chain

"""
To try out our chain, let's see what happens when we try filtering on "elenis moriset", a misspelling of Alanis Morissette, without and with retrieval:
"""
logger.info("To try out our chain, let's see what happens when we try filtering on "elenis moriset", a misspelling of Alanis Morissette, without and with retrieval:")

query = query_chain.invoke(
    {"question": "What are all the genres of elenis moriset songs", "proper_nouns": ""}
)
logger.debug(query)
db.run(query)

query = chain.invoke({"question": "What are all the genres of elenis moriset songs"})
logger.debug(query)
db.run(query)

"""
We can see that with retrieval we're able to correct the spelling from "Elenis Moriset" to "Alanis Morissette" and get back a valid result.

Another possible approach to this problem is to let an Agent decide for itself when to look up proper nouns. You can see an example of this in the [SQL: Agents](/docs/tutorials/sql_qa/#agents) guide.
"""
logger.info("We can see that with retrieval we're able to correct the spelling from "Elenis Moriset" to "Alanis Morissette" and get back a valid result.")

logger.info("\n\n[DONE]", bright=True)
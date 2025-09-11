from cassio.table.cql import STANDARD_ANALYZER
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.vectorstores import Cassandra
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnablePassthrough,
)
import cassio
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
# Hybrid Search

The standard search in LangChain is done by vector similarity. However, a number of [vector store](/docs/integrations/vectorstores/) implementations (Astra DB, ElasticSearch, Neo4J, AzureSearch, Qdrant...) also support more advanced search combining vector similarity search and other search techniques (full-text, BM25, and so on). This is generally referred to as "Hybrid" search.

**Step 1: Make sure the vectorstore you are using supports hybrid search**

At the moment, there is no unified way to perform hybrid search in LangChain. Each vectorstore may have their own way to do it. This is generally exposed as a keyword argument that is passed in during `similarity_search`.

By reading the documentation or source code, figure out whether the vectorstore you are using supports hybrid search, and, if so, how to use it.

**Step 2: Add that parameter as a configurable field for the chain**

This will let you easily call the chain and configure any relevant flags at runtime. See [this documentation](/docs/how_to/configure) for more information on configuration.

**Step 3: Call the chain with that configurable field**

Now, at runtime you can call this chain with configurable field.

## Code Example

Let's see a concrete example of what this looks like in code. We will use the Cassandra/CQL interface of Astra DB for this example.

Install the following Python package:
"""
logger.info("# Hybrid Search")

# !pip install "cassio>=0.1.7"

"""
Get the [connection secrets](https://docs.datastax.com/en/astra/astra-db-vector/get-started/quickstart.html).

Initialize cassio:
"""
logger.info(
    "Get the [connection secrets](https://docs.datastax.com/en/astra/astra-db-vector/get-started/quickstart.html).")


cassio.init(
    database_id="Your database ID",
    token="Your application token",
    keyspace="Your key space",
)

"""
Create the Cassandra VectorStore with a standard [index analyzer](https://docs.datastax.com/en/astra/astra-db-vector/cql/use-analyzers-with-cql.html). The index analyzer is needed to enable term matching.
"""
logger.info(
    "Create the Cassandra VectorStore with a standard [index analyzer](https://docs.datastax.com/en/astra/astra-db-vector/cql/use-analyzers-with-cql.html). The index analyzer is needed to enable term matching.")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Cassandra(
    embedding=embeddings,
    table_name="test_hybrid",
    body_index_options=[STANDARD_ANALYZER],
    session=None,
    keyspace=None,
)

vectorstore.add_texts(
    [
        "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",
    ]
)

"""
If we do a standard similarity search, we get all the documents:
"""
logger.info("If we do a standard similarity search, we get all the documents:")

vectorstore.as_retriever().invoke("What city did I visit last?")

"""
The Astra DB vectorstore `body_search` argument can be used to filter the search on the term `new`.
"""
logger.info(
    "The Astra DB vectorstore `body_search` argument can be used to filter the search on the term `new`.")

vectorstore.as_retriever(search_kwargs={"body_search": "new"}).invoke(
    "What city did I visit last?"
)

"""
We can now create the chain that we will use to do question-answering over
"""
logger.info(
    "We can now create the chain that we will use to do question-answering over")


"""
This is basic question-answering chain set up.
"""
logger.info("This is basic question-answering chain set up.")

template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama(model="llama3.2")

retriever = vectorstore.as_retriever()

"""
Here we mark the retriever as having a configurable field. All vectorstore retrievers have `search_kwargs` as a field. This is just a dictionary, with vectorstore specific fields
"""
logger.info("Here we mark the retriever as having a configurable field. All vectorstore retrievers have `search_kwargs` as a field. This is just a dictionary, with vectorstore specific fields")

configurable_retriever = retriever.configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

"""
We can now create the chain using our configurable retriever
"""
logger.info("We can now create the chain using our configurable retriever")

chain = (
    {"context": configurable_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("What city did I visit last?")

"""
We can now invoke the chain with configurable options. `search_kwargs` is the id of the configurable field. The value is the search kwargs to use for Astra DB.
"""
logger.info("We can now invoke the chain with configurable options. `search_kwargs` is the id of the configurable field. The value is the search kwargs to use for Astra DB.")

chain.invoke(
    "What city did I visit last?",
    config={"configurable": {"search_kwargs": {"body_search": "new"}}},
)

logger.info("\n\n[DONE]", bright=True)

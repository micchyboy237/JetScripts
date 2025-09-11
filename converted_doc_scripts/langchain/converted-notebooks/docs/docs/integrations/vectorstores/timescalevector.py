from datetime import datetime, timedelta
from dotenv import find_dotenv, load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.vectorstores.timescalevector import TimescaleVector
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from timescale_vector import client
from typing import Tuple
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
# Timescale Vector (Postgres)

>[Timescale Vector](https://www.timescale.com/ai?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral) is `PostgreSQL++` vector database for AI applications.

This notebook shows how to use the Postgres vector database `Timescale Vector`. You'll learn how to use TimescaleVector for (1) semantic search, (2) time-based vector search, (3) self-querying, and (4) how to create indexes to speed up queries.

## What is Timescale Vector?

`Timescale Vector` enables you to efficiently store and query millions of vector embeddings in `PostgreSQL`.
- Enhances `pgvector` with faster and more accurate similarity search on 100M+ vectors via `DiskANN` inspired indexing algorithm.
- Enables fast time-based vector search via automatic time-based partitioning and indexing.
- Provides a familiar SQL interface for querying vector embeddings and relational data.

`Timescale Vector` is cloud `PostgreSQL` for AI that scales with you from POC to production:
- Simplifies operations by enabling you to store relational metadata, vector embeddings, and time-series data in a single database.
- Benefits from rock-solid PostgreSQL foundation with enterprise-grade features like streaming backups and replication, high availability and row-level security.
- Enables a worry-free experience with enterprise-grade security and compliance.

## How to access Timescale Vector

`Timescale Vector` is available on [Timescale](https://www.timescale.com/ai?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral), the cloud PostgreSQL platform. (There is no self-hosted version at this time.)

LangChain users get a 90-day free trial for Timescale Vector.
- To get started, [signup](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral) to Timescale, create a new database and follow this notebook!
- See the [Timescale Vector explainer blog](https://www.timescale.com/blog/how-we-made-postgresql-the-best-vector-database/?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral) for more details and performance benchmarks.
- See the [installation instructions](https://github.com/timescale/python-vector) for more details on using Timescale Vector in Python.

## Setup

Follow these steps to get ready to follow this tutorial.
"""
logger.info("# Timescale Vector (Postgres)")

# %pip install --upgrade --quiet  timescale-vector
# %pip install --upgrade --quiet  langchain-ollama langchain-community
# %pip install --upgrade --quiet  tiktoken

"""
In this example, we'll use `OllamaEmbeddings`, so let's load your Ollama API key.
"""
logger.info(
    "In this example, we'll use `OllamaEmbeddings`, so let's load your Ollama API key.")


_ = load_dotenv(find_dotenv())
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


"""
Next we'll import the needed Python libraries and libraries from LangChain. Note that we import the `timescale-vector` library as well as the TimescaleVector LangChain vectorstore.
"""
logger.info("Next we'll import the needed Python libraries and libraries from LangChain. Note that we import the `timescale-vector` library as well as the TimescaleVector LangChain vectorstore.")


"""
## 1. Similarity Search with Euclidean Distance (Default)

First, we'll look at an example of doing a similarity search query on the State of the Union speech to find the most similar sentences to a given query sentence. We'll use the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) as our similarity metric.
"""
logger.info("## 1. Similarity Search with Euclidean Distance (Default)")

loader = TextLoader("../../../extras/modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

"""
Next, we'll load the service URL for our Timescale database. 

If you haven't already, [signup for Timescale](https://console.cloud.timescale.com/signup?utm_campaign=vectorlaunch&utm_source=langchain&utm_medium=referral), and create a new database.

Then, to connect to your PostgreSQL database, you'll need your service URI, which can be found in the cheatsheet or `.env` file you downloaded after creating a new database. 

The URI will look something like this: `postgres://tsdbadmin:<password>@<id>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require`.
"""
logger.info("Next, we'll load the service URL for our Timescale database.")

SERVICE_URL = os.environ["TIMESCALE_SERVICE_URL"]

"""
Next we create a TimescaleVector vectorstore. We specify a collection name, which will be the name of the table our data is stored in. 

Note: When creating a new instance of TimescaleVector, the TimescaleVector Module will try to create a table with the name of the collection. So, make sure that the collection name is unique (i.e it doesn't already exist).
"""
logger.info("Next we create a TimescaleVector vectorstore. We specify a collection name, which will be the name of the table our data is stored in.")

COLLECTION_NAME = "state_of_the_union_test"

db = TimescaleVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
)

"""
Now that we've loaded our data, we can perform a similarity search.
"""
logger.info("Now that we've loaded our data, we can perform a similarity search.")

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    logger.debug("-" * 80)
    logger.debug("Score: ", score)
    logger.debug(doc.page_content)
    logger.debug("-" * 80)

"""
### Using a Timescale Vector as a Retriever
After initializing a TimescaleVector store, you can use it as a [retriever](/docs/how_to#retrievers).
"""
logger.info("### Using a Timescale Vector as a Retriever")

retriever = db.as_retriever()

logger.debug(retriever)

"""
Let's look at an example of using Timescale Vector as a retriever with the RetrievalQA chain and the stuff documents chain.

In this example, we'll ask the same query as above, but this time we'll pass the relevant documents returned from Timescale Vector to an LLM to use as context to answer our question.

First we'll create our stuff chain:
"""
logger.info("Let's look at an example of using Timescale Vector as a retriever with the RetrievalQA chain and the stuff documents chain.")


llm = ChatOllama(model="llama3.2")


qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
)

query = "What did the president say about Ketanji Brown Jackson?"
response = qa_stuff.run(query)

logger.debug(response)

"""
## 2. Similarity Search with time-based filtering

A key use case for Timescale Vector is efficient time-based vector search. Timescale Vector enables this by automatically partitioning vectors (and associated metadata) by time. This allows you to efficiently query vectors by both similarity to a query vector and time.

Time-based vector search functionality is helpful for applications like:
- Storing and retrieving LLM response history (e.g. chatbots)
- Finding the most recent embeddings that are similar to a query vector (e.g recent news).
- Constraining similarity search to a relevant time range (e.g asking time-based questions about a knowledge base)

To illustrate how to use TimescaleVector's time-based vector search functionality, we'll ask questions about the git log history for TimescaleDB . We'll illustrate how to add documents with a time-based uuid and how run similarity searches with time range filters.

### Extract content and metadata from git log JSON
First lets load in the git log data into a new collection in our PostgreSQL database named `timescale_commits`.
"""
logger.info("## 2. Similarity Search with time-based filtering")


"""
W
e
'
l
l
 
d
e
f
i
n
e
 
a
 
h
e
l
p
e
r
 
f
u
n
c
t
i
o
n
 
t
o
 
c
r
e
a
t
e
 
a
 
u
u
i
d
 
f
o
r
 
a
 
d
o
c
u
m
e
n
t
 
a
n
d
 
a
s
s
o
c
i
a
t
e
d
 
v
e
c
t
o
r
 
e
m
b
e
d
d
i
n
g
 
b
a
s
e
d
 
o
n
 
i
t
s
 
t
i
m
e
s
t
a
m
p
.
 
W
e
'
l
l
 
u
s
e
 
t
h
i
s
 
f
u
n
c
t
i
o
n
 
t
o
 
c
r
e
a
t
e
 
a
 
u
u
i
d
 
f
o
r
 
e
a
c
h
 
g
i
t
 
l
o
g
 
e
n
t
r
y
.


I
m
p
o
r
t
a
n
t
 
n
o
t
e
:
 
I
f
 
y
o
u
 
a
r
e
 
w
o
r
k
i
n
g
 
w
i
t
h
 
d
o
c
u
m
e
n
t
s
 
a
n
d
 
w
a
n
t
 
t
h
e
 
c
u
r
r
e
n
t
 
d
a
t
e
 
a
n
d
 
t
i
m
e
 
a
s
s
o
c
i
a
t
e
d
 
w
i
t
h
 
v
e
c
t
o
r
 
f
o
r
 
t
i
m
e
-
b
a
s
e
d
 
s
e
a
r
c
h
,
 
y
o
u
 
c
a
n
 
s
k
i
p
 
t
h
i
s
 
s
t
e
p
.
 
A
 
u
u
i
d
 
w
i
l
l
 
b
e
 
a
u
t
o
m
a
t
i
c
a
l
l
y
 
g
e
n
e
r
a
t
e
d
 
w
h
e
n
 
t
h
e
 
d
o
c
u
m
e
n
t
s
 
a
r
e
 
i
n
g
e
s
t
e
d
 
b
y
 
d
e
f
a
u
l
t
.
"""
logger.info("W")


def create_uuid(date_string: str):
    if date_string is None:
        return None
    time_format = "%a %b %d %H:%M:%S %Y %z"
    datetime_obj = datetime.strptime(date_string, time_format)
    uuid = client.uuid_from_time(datetime_obj)
    return str(uuid)


"""
Next, we'll define a metadata function to extract the relevant metadata from the JSON record. We'll pass this function to the JSONLoader. See the [JSON document loader docs](/docs/how_to/document_loader_json) for more details.
"""
logger.info(
    "Next, we'll define a metadata function to extract the relevant metadata from the JSON record. We'll pass this function to the JSONLoader. See the [JSON document loader docs](/docs/how_to/document_loader_json) for more details.")


def split_name(input_string: str) -> Tuple[str, str]:
    if input_string is None:
        return None, None
    start = input_string.find("<")
    end = input_string.find(">")
    name = input_string[:start].strip()
    email = input_string[start + 1: end].strip()
    return name, email


def create_date(input_string: str) -> datetime:
    if input_string is None:
        return None
    month_dict = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }

    components = input_string.split()
    day = components[2]
    month = month_dict[components[1]]
    year = components[4]
    time = components[3]
    # Convert the offset to minutes
    timezone_offset_minutes = int(components[5])
    timezone_hours = timezone_offset_minutes // 60  # Calculate the hours
    timezone_minutes = timezone_offset_minutes % 60  # Calculate the remaining minutes
    timestamp_tz_str = (
        f"{year}-{month}-{day} {time}+{timezone_hours:02}{timezone_minutes:02}"
    )
    return timestamp_tz_str


def extract_metadata(record: dict, metadata: dict) -> dict:
    record_name, record_email = split_name(record["author"])
    metadata["id"] = create_uuid(record["date"])
    metadata["date"] = create_date(record["date"])
    metadata["author_name"] = record_name
    metadata["author_email"] = record_email
    metadata["commit_hash"] = record["commit"]
    return metadata


"""
Next, you'll need to [download the sample dataset](https://s3.amazonaws.com/assets.timescale.com/ai/ts_git_log.json) and place it in the same directory as this notebook.

You can use following command:
"""
logger.info(
    "Next, you'll need to [download the sample dataset](https://s3.amazonaws.com/assets.timescale.com/ai/ts_git_log.json) and place it in the same directory as this notebook.")

# !curl -O https://s3.amazonaws.com/assets.timescale.com/ai/ts_git_log.json

"""
Finally we can initialize the JSON loader to parse the JSON records. We also remove empty records for simplicity.
"""
logger.info("Finally we can initialize the JSON loader to parse the JSON records. We also remove empty records for simplicity.")

FILE_PATH = "../../../../../ts_git_log.json"

loader = JSONLoader(
    file_path=FILE_PATH,
    jq_schema=".commit_history[]",
    text_content=False,
    metadata_func=extract_metadata,
)
documents = loader.load()

documents = [doc for doc in documents if doc.metadata["date"] is not None]

logger.debug(documents[0])

"""
### Load documents and metadata into TimescaleVector vectorstore
Now that we have prepared our documents, let's process them and load them, along with their vector embedding representations into our TimescaleVector vectorstore.

Since this is a demo, we will only load the first 500 records. In practice, you can load as many records as you want.
"""
logger.info("### Load documents and metadata into TimescaleVector vectorstore")

NUM_RECORDS = 500
documents = documents[:NUM_RECORDS]

"""
Then we use the CharacterTextSplitter to split the documents into smaller chunks if needed for easier embedding. Note that this splitting process retains the metadata for each document.
"""
logger.info("Then we use the CharacterTextSplitter to split the documents into smaller chunks if needed for easier embedding. Note that this splitting process retains the metadata for each document.")

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)

"""
Next we'll create a Timescale Vector instance from the collection of documents that we finished pre-processsing.

First, we'll define a collection name, which will be the name of our table in the PostgreSQL database. 

We'll also define a time delta, which we pass to the `time_partition_interval` argument, which will be used to as the interval for partitioning the data by time. Each partition will consist of data for the specified length of time. We'll use 7 days for simplicity, but you can pick whatever value make sense for your use case -- for example if you query recent vectors frequently you might want to use a smaller time delta like 1 day, or if you query vectors over a decade long time period then you might want to use a larger time delta like 6 months or 1 year.

Finally, we'll create the TimescaleVector instance. We specify the `ids` argument to be the `uuid` field in our metadata that we created in the pre-processing step above. We do this because we want the time part of our uuids to reflect dates in the past (i.e when the commit was made). However, if we wanted the current date and time to be associated with our document, we can remove the id argument and uuid's will be automatically created with the current date and time.
"""
logger.info("Next we'll create a Timescale Vector instance from the collection of documents that we finished pre-processsing.")

COLLECTION_NAME = "timescale_commits"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = TimescaleVector.from_documents(
    embedding=embeddings,
    ids=[doc.metadata["id"] for doc in docs],
    documents=docs,
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
    time_partition_interval=timedelta(days=7),
)

"""
### Querying vectors by time and similarity

Now that we have loaded our documents into TimescaleVector, we can query them by time and similarity.

TimescaleVector provides multiple methods for querying vectors by doing similarity search with time-based filtering.

Let's take a look at each method below:
"""
logger.info("### Querying vectors by time and similarity")

# Start date = 1 August 2023, 22:10:35
start_dt = datetime(2023, 8, 1, 22, 10, 35)
# End date = 30 August 2023, 22:10:35
end_dt = datetime(2023, 8, 30, 22, 10, 35)
td = timedelta(days=7)  # Time delta = 7 days

query = "What's new with TimescaleDB functions?"

"""
Method 1: Filter within a provided start date and end date.
"""
logger.info("Method 1: Filter within a provided start date and end date.")

docs_with_score = db.similarity_search_with_score(
    query, start_date=start_dt, end_date=end_dt
)

for doc, score in docs_with_score:
    logger.debug("-" * 80)
    logger.debug("Score: ", score)
    logger.debug("Date: ", doc.metadata["date"])
    logger.debug(doc.page_content)
    logger.debug("-" * 80)

"""
Note how the query only returns results within the specified date range.

Method 2: Filter within a provided start date, and a time delta later.
"""
logger.info(
    "Note how the query only returns results within the specified date range.")

docs_with_score = db.similarity_search_with_score(
    query, start_date=start_dt, time_delta=td
)

for doc, score in docs_with_score:
    logger.debug("-" * 80)
    logger.debug("Score: ", score)
    logger.debug("Date: ", doc.metadata["date"])
    logger.debug(doc.page_content)
    logger.debug("-" * 80)

"""
Once again, notice how we get results within the specified time filter, different from the previous query.

Method 3: Filter within a provided end date and a time delta earlier.
"""
logger.info("Once again, notice how we get results within the specified time filter, different from the previous query.")

docs_with_score = db.similarity_search_with_score(
    query, end_date=end_dt, time_delta=td)

for doc, score in docs_with_score:
    logger.debug("-" * 80)
    logger.debug("Score: ", score)
    logger.debug("Date: ", doc.metadata["date"])
    logger.debug(doc.page_content)
    logger.debug("-" * 80)

"""
Method 4: We can also filter for all vectors after a given date by only specifying a start date in our query.

Method 5: Similarly, we can filter for or all vectors before a given date by only specify an end date in our query.
"""
logger.info("Method 4: We can also filter for all vectors after a given date by only specifying a start date in our query.")

docs_with_score = db.similarity_search_with_score(query, start_date=start_dt)

for doc, score in docs_with_score:
    logger.debug("-" * 80)
    logger.debug("Score: ", score)
    logger.debug("Date: ", doc.metadata["date"])
    logger.debug(doc.page_content)
    logger.debug("-" * 80)

docs_with_score = db.similarity_search_with_score(query, end_date=end_dt)

for doc, score in docs_with_score:
    logger.debug("-" * 80)
    logger.debug("Score: ", score)
    logger.debug("Date: ", doc.metadata["date"])
    logger.debug(doc.page_content)
    logger.debug("-" * 80)

"""
The main takeaway is that in each result above, only vectors within the specified time range are returned. These queries are very efficient as they only need to search the relevant partitions.

We can also use this functionality for question answering, where we want to find the most relevant vectors within a specified time range to use as context for answering a question. Let's take a look at an example below, using Timescale Vector as a retriever:
"""
logger.info("The main takeaway is that in each result above, only vectors within the specified time range are returned. These queries are very efficient as they only need to search the relevant partitions.")

retriever = db.as_retriever(
    search_kwargs={"start_date": start_dt, "end_date": end_dt})


llm = ChatOllama(model="llama3.2")


qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
)

query = (
    "What's new with the timescaledb functions? Tell me when these changes were made."
)
response = qa_stuff.run(query)
logger.debug(response)

"""
Note that the context the LLM uses to compose an answer are from retrieved documents only within the specified date range. 

This shows how you can use Timescale Vector to enhance retrieval augmented generation by retrieving documents within time ranges relevant to your query.

## 3. Using ANN Search Indexes to Speed Up Queries

You can speed up similarity queries by creating an index on the embedding column. You should only do this once you have ingested a large part of your data.

Timescale Vector supports the following indexes:
- timescale_vector index (tsv): a disk-ann inspired graph index for fast similarity search (default).
- pgvector's HNSW index: a hierarchical navigable small world graph index for fast similarity search.
- pgvector's IVFFLAT index: an inverted file index for fast similarity search.

Important note: In PostgreSQL, each table can only have one index on a particular column. So if you'd like to test the performance of different index types, you can do so either by (1) creating multiple tables with different indexes, (2) creating multiple vector columns in the same table and creating different indexes on each column, or (3) by dropping and recreating the index on the same column and comparing results.
"""
logger.info("## 3. Using ANN Search Indexes to Speed Up Queries")

COLLECTION_NAME = "timescale_commits"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = TimescaleVector(
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
    embedding_function=embeddings,
)

"""
Using the `create_index()` function without additional arguments will create a timescale_vector_index by default, using the default parameters.
"""
logger.info("Using the `create_index()` function without additional arguments will create a timescale_vector_index by default, using the default parameters.")

db.create_index()

"""
You can also specify the parameters for the index. See the Timescale Vector documentation for a full discussion of the different parameters and their effects on performance.

Note: You don't need to specify parameters as we set smart defaults. But you can always specify your own parameters if you want to experiment eek out more performance for your specific dataset.
"""
logger.info("You can also specify the parameters for the index. See the Timescale Vector documentation for a full discussion of the different parameters and their effects on performance.")

db.drop_index()

db.create_index(index_type="tsv", max_alpha=1.0, num_neighbors=50)

"""
Timescale Vector also supports the HNSW ANN indexing algorithm, as well as the ivfflat ANN indexing algorithm. Simply specify in the `index_type` argument which index you'd like to create, and optionally specify the parameters for the index.
"""
logger.info("Timescale Vector also supports the HNSW ANN indexing algorithm, as well as the ivfflat ANN indexing algorithm. Simply specify in the `index_type` argument which index you'd like to create, and optionally specify the parameters for the index.")

db.drop_index()

db.create_index(index_type="hnsw", m=16, ef_construction=64)

db.drop_index()

db.create_index(index_type="ivfflat", num_lists=20, num_records=1000)

"""
In general, we recommend using the default timescale vector index, or the HNSW index.
"""
logger.info(
    "In general, we recommend using the default timescale vector index, or the HNSW index.")

db.drop_index()
db.create_index()

"""
## 4. Self Querying Retriever with Timescale Vector

Timescale Vector also supports the self-querying retriever functionality, which gives it the ability to query itself. Given a natural language query with a query statement and filters (single or composite), the retriever uses a query constructing LLM chain to write a SQL query and then applies it to the underlying PostgreSQL database in the Timescale Vector vectorstore.

For more on self-querying, [see the docs](/docs/how_to/self_query).

To illustrate self-querying with Timescale Vector, we'll use the same gitlog dataset from Part 3.
"""
logger.info("## 4. Self Querying Retriever with Timescale Vector")

COLLECTION_NAME = "timescale_commits"
vectorstore = TimescaleVector(
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
)

"""
Next we'll create our self-querying retriever. To do this we'll need to provide some information upfront about the metadata fields that our documents support and a short description of the document contents.
"""
logger.info("Next we'll create our self-querying retriever. To do this we'll need to provide some information upfront about the metadata fields that our documents support and a short description of the document contents.")


metadata_field_info = [
    AttributeInfo(
        name="id",
        description="A UUID v1 generated from the date of the commit",
        type="uuid",
    ),
    AttributeInfo(
        name="date",
        description="The date of the commit in timestamptz format",
        type="timestamptz",
    ),
    AttributeInfo(
        name="author_name",
        description="The name of the author of the commit",
        type="string",
    ),
    AttributeInfo(
        name="author_email",
        description="The email address of the author of the commit",
        type="string",
    ),
]
document_content_description = "The git log commit summary containing the commit hash, author, date of commit, change summary and change details"

llm = ChatOllama(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)

"""
Now let's test out the self-querying retriever on our gitlog dataset. 

Run the queries below and note how you can specify a query, query with a filter, and query with a composite filter (filters with AND, OR) in natural language and the self-query retriever will translate that query into SQL and perform the search on the Timescale Vector PostgreSQL vectorstore.

This illustrates the power of the self-query retriever. You can use it to perform complex searches over your vectorstore without you or your users having to write any SQL directly!
"""
logger.info(
    "Now let's test out the self-querying retriever on our gitlog dataset.")

retriever.invoke("What are improvements made to continuous aggregates?")

retriever.invoke("What commits did Sven Klemm add?")

retriever.invoke(
    "What commits about timescaledb_functions did Sven Klemm add?")

retriever.invoke("What commits were added in July 2023?")

retriever.invoke(
    "What are two commits about hierarchical continuous aggregates?")

"""
## 5. Working with an existing TimescaleVector vectorstore

In the examples above, we created a vectorstore from a collection of documents. However, often we want to work insert data into and query data from an existing vectorstore. Let's see how to initialize, add documents to, and query an existing collection of documents in a TimescaleVector vector store.

To work with an existing Timescale Vector store, we need to know the name of the table we want to query (`COLLECTION_NAME`) and the URL of the cloud PostgreSQL database (`SERVICE_URL`).
"""
logger.info("## 5. Working with an existing TimescaleVector vectorstore")

COLLECTION_NAME = "timescale_commits"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = TimescaleVector(
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
    embedding_function=embeddings,
)

"""
To load new data into the table, we use the `add_document()` function. This function takes a list of documents and a list of metadata. The metadata must contain a unique id for each document. 

If you want your documents to be associated with the current date and time, you do not need to create a list of ids. A uuid will be automatically generated for each document.

If you want your documents to be associated with a past date and time, you can create a list of ids using the `uuid_from_time` function in the `timecale-vector` python library, as shown in Section 2 above. This function takes a datetime object and returns a uuid with the date and time encoded in the uuid.
"""
logger.info("To load new data into the table, we use the `add_document()` function. This function takes a list of documents and a list of metadata. The metadata must contain a unique id for each document.")

ids = vectorstore.add_documents([Document(page_content="foo")])
ids

docs_with_score = vectorstore.similarity_search_with_score("foo")

docs_with_score[0]

docs_with_score[1]

"""
### Deleting Data 

You can delete data by uuid or by a filter on the metadata.
"""
logger.info("### Deleting Data")

ids = vectorstore.add_documents([Document(page_content="Bar")])

vectorstore.delete(ids)

"""
Deleting using metadata is especially useful if you want to periodically update information scraped from a particular source, or particular date or some other metadata attribute.
"""
logger.info("Deleting using metadata is especially useful if you want to periodically update information scraped from a particular source, or particular date or some other metadata attribute.")

vectorstore.add_documents(
    [Document(page_content="Hello World", metadata={
              "source": "www.example.com/hello"})]
)
vectorstore.add_documents(
    [Document(page_content="Adios", metadata={
              "source": "www.example.com/adios"})]
)

vectorstore.delete_by_metadata({"source": "www.example.com/adios"})

vectorstore.add_documents(
    [
        Document(
            page_content="Adios, but newer!",
            metadata={"source": "www.example.com/adios"},
        )
    ]
)

"""
### Overriding a vectorstore

If you have an existing collection, you override it by doing `from_documents` and setting `pre_delete_collection` = True
"""
logger.info("### Overriding a vectorstore")

db = TimescaleVector.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    service_url=SERVICE_URL,
    pre_delete_collection=True,
)

docs_with_score = db.similarity_search_with_score("foo")

docs_with_score[0]

logger.info("\n\n[DONE]", bright=True)

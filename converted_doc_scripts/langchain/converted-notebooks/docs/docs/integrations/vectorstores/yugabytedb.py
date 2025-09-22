from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from langchain_yugabytedb import YBEngine, YugabyteDBVectorStore
from langchain_yugabytedb import YugabyteDBChatMessageHistory
import os
import psycopg
import shutil
import uuid


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
---
sidebar_label: YugabyteDB
---

# YugabyteDBVectorStore

This notebook covers how to get started with the YugabyteDB vector store in langchain, using the `langchain-yugabytedb` package.

YugabyteDB is a cloud-native distributed PostgreSQL-compatible database that combines strong consistency with ultra-resilience, seamless scalability, geo-distribution, and highly flexible data locality to deliver business-critical, transactional applications.

[YugabyteDB](https://www.yugabyte.com/ai/) combines the power of the `pgvector` PostgreSQL extension with an inherently distributed architecture. This future-proofed foundation helps you build GenAI applications using RAG retrieval that demands high-performance vector search.

YugabyteDB’s unique approach to vector indexing addresses the limitations of single-node PostgreSQL systems when dealing with large-scale vector datasets.


## Setup

### Minimum Version
`langchain-yugabytedb` module requires YugabyteDB `v2025.1.0.0` or higher.

### Connecting to YugabyteDB database

In order to get started with `YugabyteDBVectorStore`, lets start a local YugabyteDB node for development purposes - 

### Start YugabyteDB RF-1 Universe.
"""
logger.info("# YugabyteDBVectorStore")

docker run -d --name yugabyte_node01 --hostname yugabyte01 \
  -p 7000:7000 -p 9000:9000 -p 15433:15433 -p 5433:5433 -p 9042:9042 \
  yugabytedb/yugabyte:2.25.2.0-b359 bin/yugabyted start --background=false \
  --master_flags="allowed_preview_flags_csv=ysql_yb_enable_advisory_locks,ysql_yb_enable_advisory_locks=true" \
  --tserver_flags="allowed_preview_flags_csv=ysql_yb_enable_advisory_locks,ysql_yb_enable_advisory_locks=true"

docker exec -it yugabyte_node01 bin/ysqlsh -h yugabyte01 -c "CREATE extension vector;"

"""
For production deployment, performance benchmarking, or deploying a true multi-node on multi-host setup, see Deploy [YugabyteDB](https://docs.yugabyte.com/stable/deploy/).

## Installation
"""
logger.info("## Installation")

# %pip install --upgrade --quiet  langchain
# %pip install --upgrade --quiet  langchain-ollama langchain-community tiktoken
# %pip install --upgrade --quiet  psycopg-binary

# %pip install -qU "langchain-yugabytedb"

"""
### Set your YugabyteDB Values

YugabyteDB clients connect to the cluster using a PostgreSQL compliant connection string. YugabyteDB connection parameters are provided below.
"""
logger.info("### Set your YugabyteDB Values")

YUGABYTEDB_USER = "yugabyte"  # @param {type: "string"}
YUGABYTEDB_PASSWORD = ""  # @param {type: "string"}
YUGABYTEDB_HOST = "localhost"  # @param {type: "string"}
YUGABYTEDB_PORT = "5433"  # @param {type: "string"}
YUGABYTEDB_DB = "yugabyte"  # @param {type: "string"}

"""
## Initialization

### Environment Setup

# This notebook uses the Ollama API through `OllamaEmbeddings`. We suggest obtaining an Ollama API key and export it as an environment variable with the name `OPENAI_API_KEY`.

### Connecting to YugabyteDB Universe
"""
logger.info("## Initialization")


TABLE_NAME = "my_doc_collection"
VECTOR_SIZE = 1536

CONNECTION_STRING = (
    f"postgresql+asyncpg://{YUGABYTEDB_USER}:{YUGABYTEDB_PASSWORD}@{YUGABYTEDB_HOST}"
    f":{YUGABYTEDB_PORT}/{YUGABYTEDB_DB}"
)
engine = YBEngine.from_connection_string(url=CONNECTION_STRING)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
engine.init_vectorstore_table(
    table_name=TABLE_NAME,
    vector_size=VECTOR_SIZE,
)

yugabyteDBVectorStore = YugabyteDBVectorStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
    embedding_service=embeddings,
)

"""
## Manage vector store

### Add items to vector store
"""
logger.info("## Manage vector store")


docs = [
    Document(page_content="Apples and oranges"),
    Document(page_content="Cars and airplanes"),
    Document(page_content="Train"),
]

yugabyteDBVectorStore.add_documents(docs)

"""
['b40e7f47-3a4e-4b88-b6e2-cb3465dde6bd', '275823d2-1a47-440d-904b-c07b132fd72b', 'f0c5a9bc-1456-40fe-906b-4e808d601470']

### Delete items from vector store
"""
logger.info("### Delete items from vector store")

yugabyteDBVectorStore.delete(ids=["275823d2-1a47-440d-904b-c07b132fd72b"])

"""
### Update items from vector store

Note: Update operation is not supported by YugabyteDBVectorStore.

## Query vector store

Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent. 

### Query directly

Performing a simple similarity search can be done as follows:
"""
logger.info("### Update items from vector store")

query = "I'd like a fruit."
docs = yugabyteDBVectorStore.similarity_search(query)
logger.debug(docs)

"""
If you want to execute a similarity search and receive the corresponding scores you can run:
"""
logger.info("If you want to execute a similarity search and receive the corresponding scores you can run:")

query = "I'd like a fruit."
docs = yugabyteDBVectorStore.similarity_search(query, k=1)
logger.debug(docs)

"""
### Query by turning into retriever

You can also transform the vector store into a retriever for easier usage in your chains.
"""
logger.info("### Query by turning into retriever")

retriever = yugabyteDBVectorStore.as_retriever(search_kwargs={"k": 1})
retriever.invoke("I'd like a fruit.")

"""
## ChatMessageHistory

The chat message history abstraction helps to persist chat message history in a YugabyteDB table.

`YugabyteDBChatMessageHistory` is parameterized using a table_name and a session_id.

The table_name is the name of the table in the database where the chat messages will be stored.

The session_id is a unique identifier for the chat session. It can be assigned by the caller using uuid.uuid4()
"""
logger.info("## ChatMessageHistory")



conn_info = "dbname=yugabyte user=yugabyte host=localhost port=5433"
sync_connection = psycopg.connect(conn_info)

table_name = "chat_history"
YugabyteDBChatMessageHistory.create_tables(sync_connection, table_name)

session_id = str(uuid.uuid4())

chat_history = YugabyteDBChatMessageHistory(
    table_name, session_id, sync_connection=sync_connection
)

chat_history.add_messages(
    [
        SystemMessage(content="Meow"),
        AIMessage(content="woof"),
        HumanMessage(content="bark"),
    ]
)

logger.debug(chat_history.messages)

"""
## Usage for retrieval-augmented generation


One of the primary advantages of the vector stores is to provide contextual data to the LLMs. LLMs often are trained with stale data and might not have the relevant domain specific knowledge which results in halucinations in LLMs responses. Take the following example -
"""
logger.info("## Usage for retrieval-augmented generation")

# import getpass

# my_api_key = getpass.getpass("Enter your API Key: ")

llm = ChatOllama(model="llama3.2")
messages = [
    SystemMessage(
        content="You are a helpful and friendly assistant named 'YugaAI'. You love to answer questions about YugabyteDB and distributed sql."
    ),
    HumanMessage(content="Hi YugaAI! Where's the headquarters of YugabyteDB?"),
]

logger.debug("--- First Interaction ---")
logger.debug(f"Human: {messages[1].content}")  # Print the human message
response1 = llm.invoke(messages)
logger.debug(f"YugaAI: {response1.content}")

logger.debug("\n--- Second Interaction ---")
logger.debug(f"Human: {messages[2].content}")  # Print the new human message
response2 = llm.invoke(messages)  # Send the *entire* message history
logger.debug(f"YugaAI: {response2.content}")

messages.append(AIMessage(content=response2.content))

messages.append(
    HumanMessage(
        content="Can you tell me the current preview release version of YugabyteDB?"
    )
)

logger.debug("\n--- Third Interaction ---")
logger.debug(f"Human: {messages[4].content}")  # Print the new human message
response3 = llm.invoke(messages)  # Send the *entire* message history
logger.debug(f"YugaAI: {response3.content}")

--- First Interaction ---
Human: Hi YugaAI! Where's the headquarters of YugabyteDB?
YugaAI: Hello! YugabyteDB's headquarters is located in Sunnyvale, California, USA.

--- Second Interaction ---
Human: And what are YugabyteDB's supported APIs?
YugaAI: YugabyteDB's headquarters is located in Sunnyvale, California, USA.

YugabyteDB supports several APIs, including:
1. YSQL (PostgreSQL-compatible SQL)
2. YCQL (Cassandra-compatible query language)
3. YEDIS (Redis-compatible key-value store)

These APIs allow developers to interact with YugabyteDB using familiar interfaces and tools.

--- Third Interaction ---
Human: Can you tell me the current preview release version of YugabyteDB?
YugaAI: The current preview release version of YugabyteDB is 2.11.0. This version includes new features, improvements, and bug fixes that are being tested by the community before the official stable release.

"""
The current preview release of YugabyteDB is `v2.25.2.0`, however LLMs is providing stale information which is 2-3 years old. This is where the vector stores complement the LLMs by providing a way to store and retrive relevant information.

### Construct a RAG for providing contextual information

We will provide the relevant information to the LLMs by reading the YugabyteDB documentation. Let's first read the YugabyteDB docs and add data into YugabyteDB Vectorstore by loading, splitting and chuncking data from a html source. We will then store the vector embeddings generated by Ollama embeddings into YugabyteDB Vectorstore.

#### Generate Embeddings
"""
logger.info("### Construct a RAG for providing contextual information")

# import getpass

# my_api_key = getpass.getpass("Enter your API Key: ")
url = "https://docs.yugabyte.com/preview/releases/ybdb-releases/v2.25/"

loader = WebBaseLoader(url)

documents = loader.load()

logger.debug(f"Number of documents loaded: {len(documents)}")

for i, doc in enumerate(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",  # Split by double newline (common paragraph separator)
        chunk_size=1000,  # Each chunk will aim for 1000 characters
        chunk_overlap=200,  # Allow 200 characters overlap between chunks
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    logger.debug(f"\n--- After Splitting ({len(chunks)} chunks) ---")

    CONNECTION_STRING = "postgresql+psycopg://yugabyte:@localhost:5433/yugabyte"
    TABLE_NAME = "yb_relnotes_chunks"
    VECTOR_SIZE = 1536
    engine = YBEngine.from_connection_string(url=CONNECTION_STRING)
    engine.init_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=VECTOR_SIZE,
    )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = YugabyteDBVectorStore.from_documents(
        engine=engine, table_name=TABLE_NAME, documents=chunks, embedding=embeddings
    )

    logger.debug(f"Successfully stored {len(chunks)} chunks in PostgreSQL table: {TABLE_NAME}")

"""
#### Configure the YugabyteDB retriever
"""
logger.info("#### Configure the YugabyteDB retriever")


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
logger.debug(
    f"Retriever created, set to retrieve top {retriever.search_kwargs['k']} documents."
)


llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and friendly assistant named 'YugaAI'. You love to answer questions about YugabyteDB and distributed sql.",
        ),
        ("human", "Context: {context}\nQuestion: {question}"),
    ]
)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

"""
Now, let's try asking the same question `Can you tell me the current preview release version of YugabyteDB?` again to the LLM
"""
logger.info("Now, let's try asking the same question `Can you tell me the current preview release version of YugabyteDB?` again to the LLM")

rag_query = "Can you tell me the current preview release version of YugabyteDB?"
logger.debug(f"\nQuerying RAG chain: '{rag_query}'")
rag_response = rag_chain.invoke(rag_query)
logger.debug("\n--- RAG Chain Response ---")
logger.debug(rag_response)

"""
Querying RAG chain: 'Can you tell me the current preview release version of YugabyteDB?'
"""
logger.info("Querying RAG chain: 'Can you tell me the current preview release version of YugabyteDB?'")

--- RAG Chain Response ---
The current preview release version of YugabyteDB is v2.25.2.0.

"""
## API reference
    
For detailed information of all YugabyteDBVectorStore features and configurations head to the langchain-yugabytedb github repo: https://github.com/yugabyte/langchain-yugabytedb"
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
import asyncio
from jet.transformers.formatters import format_json
from google.colab import auth
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex
from llama_index_alloydb_pg import AlloyDBChatStore
from llama_index_alloydb_pg import AlloyDBEngine
from llama_index_alloydb_pg import AlloyDBVectorStore
import google.auth
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Google AlloyDB for PostgreSQL - `AlloyDBChatStore`

> [AlloyDB](https://cloud.google.com/alloydb) is a fully managed relational database service that offers high performance, seamless integration, and impressive scalability. AlloyDB is 100% compatible with PostgreSQL. Extend your database application to build AI-powered experiences leveraging AlloyDB's LlamaIndex integrations.

This notebook goes over how to use `AlloyDB for PostgreSQL` to store chat history with `AlloyDBChatStore` class.

Learn more about the package on [GitHub](https://github.com/googleapis/llama-index-alloydb-pg-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/llama-index-alloydb-pg-python/blob/main/samples/llama_index_chat_store.ipynb)

## Before you begin

To run this notebook, you will need to do the following:

 * [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
 * [Enable the AlloyDB API](https://console.cloud.google.com/flows/enableapi?apiid=alloydb.googleapis.com)
 * [Create a AlloyDB cluster and instance.](https://cloud.google.com/alloydb/docs/cluster-create)
 * [Create a AlloyDB database.](https://cloud.google.com/alloydb/docs/quickstart/create-and-connect)
 * [Add a User to the database.](https://cloud.google.com/alloydb/docs/database-users/about)

### 🦙 Library Installation
Install the integration library, `llama-index-alloydb-pg`, and the library for the embedding service, `llama-index-embeddings-vertex`.
"""
logger.info("# Google AlloyDB for PostgreSQL - `AlloyDBChatStore`")

# %pip install --upgrade --quiet llama-index-alloydb-pg llama-index-llms-vertex llama-index

"""
**Colab only:** Uncomment the following cell to restart the kernel or use the button to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.
"""



"""
### 🔐 Authentication
Authenticate to Google Cloud as the IAM user logged into this notebook in order to access your Google Cloud Project.

* If you are using Colab to run this notebook, use the cell below and continue.
* If you are using Vertex AI Workbench, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).
"""
logger.info("### 🔐 Authentication")


auth.authenticate_user()

"""
### ☁ Set Your Google Cloud Project
Set your Google Cloud project so that you can leverage Google Cloud resources within this notebook.

If you don't know your project ID, try the following:

* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113).
"""
logger.info("### ☁ Set Your Google Cloud Project")

PROJECT_ID = "my-project-id"  # @param {type:"string"}

# !gcloud config set project {PROJECT_ID}

"""
## Basic Usage

### Set AlloyDB database values
Find your database values, in the [AlloyDB Instances page](https://console.cloud.google.com/alloydb/clusters).
"""
logger.info("## Basic Usage")

REGION = "us-central1"  # @param {type: "string"}
CLUSTER = "my-cluster"  # @param {type: "string"}
INSTANCE = "my-primary"  # @param {type: "string"}
DATABASE = "my-database"  # @param {type: "string"}
TABLE_NAME = "chat_store"  # @param {type: "string"}
VECTOR_STORE_TABLE_NAME = "vector_store"  # @param {type: "string"}
USER = "postgres"  # @param {type: "string"}
PASSWORD = "my-password"  # @param {type: "string"}

"""
### AlloyDBEngine Connection Pool

One of the requirements and arguments to establish AlloyDB as a chat store is a `AlloyDBEngine` object. The `AlloyDBEngine`  configures a connection pool to your AlloyDB database, enabling successful connections from your application and following industry best practices.

To create a `AlloyDBEngine` using `AlloyDBEngine.from_instance()` you need to provide only 5 things:

1. `project_id` : Project ID of the Google Cloud Project where the AlloyDB instance is located.
1. `region` : Region where the AlloyDB instance is located.
1. `cluster`: The name of the AlloyDB cluster.
1. `instance` : The name of the AlloyDB instance.
1. `database` : The name of the database to connect to on the AlloyDB instance.

By default, [IAM database authentication](https://cloud.google.com/alloydb/docs/connect-iam) will be used as the method of database authentication. This library uses the IAM principal belonging to the [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials) sourced from the environment.

Optionally, [built-in database authentication](https://cloud.google.com/alloydb/docs/database-users/about) using a username and password to access the AlloyDB database can also be used. Just provide the optional `user` and `password` arguments to `AlloyDBEngine.from_instance()`:

* `user` : Database user to use for built-in database authentication and login
* `password` : Database password to use for built-in database authentication and login.

**Note:** This tutorial demonstrates the async interface. All async methods have corresponding sync methods.
"""
logger.info("### AlloyDBEngine Connection Pool")


async def async_func_2():
    engine = await AlloyDBEngine.afrom_instance(
        project_id=PROJECT_ID,
        region=REGION,
        cluster=CLUSTER,
        instance=INSTANCE,
        database=DATABASE,
        user=USER,
        password=PASSWORD,
    )
    return engine
engine = asyncio.run(async_func_2())
logger.success(format_json(engine))

"""
### AlloyDBEngine for AlloyDB Omni
To create an `AlloyDBEngine` for AlloyDB Omni, you will need a connection url. `AlloyDBEngine.from_connection_string` first creates an async engine and then turns it into an `AlloyDBEngine`. Here is an example connection with the `asyncpg` driver:
"""
logger.info("### AlloyDBEngine for AlloyDB Omni")

OMNI_USER = "my-omni-user"
OMNI_PASSWORD = ""
OMNI_HOST = "127.0.0.1"
OMNI_PORT = "5432"
OMNI_DATABASE = "my-omni-db"

connstring = f"postgresql+asyncpg://{OMNI_USER}:{OMNI_PASSWORD}@{OMNI_HOST}:{OMNI_PORT}/{OMNI_DATABASE}"
engine = AlloyDBEngine.from_connection_string(connstring)

"""
### Initialize a table
The `AlloyDBChatStore` class requires a database table. The `AlloyDBEngine` engine has a helper method `ainit_chat_store_table()` that can be used to create a table with the proper schema for you.
"""
logger.info("### Initialize a table")

async def run_async_code_f27d43d4():
    await engine.ainit_chat_store_table(table_name=TABLE_NAME)
    return 
 = asyncio.run(run_async_code_f27d43d4())
logger.success(format_json())

"""
#### Optional Tip: 💡
You can also specify a schema name by passing `schema_name` wherever you pass `table_name`.
"""
logger.info("#### Optional Tip: 💡")

SCHEMA_NAME = "my_schema"

await engine.ainit_chat_store_table(
    table_name=TABLE_NAME,
    schema_name=SCHEMA_NAME,
)

"""
### Initialize a default AlloyDBChatStore
"""
logger.info("### Initialize a default AlloyDBChatStore")


async def async_func_2():
    chat_store = await AlloyDBChatStore.create(
        engine=engine,
        table_name=TABLE_NAME,
    )
    return chat_store
chat_store = asyncio.run(async_func_2())
logger.success(format_json(chat_store))

"""
### Create a ChatMemoryBuffer
"""
logger.info("### Create a ChatMemoryBuffer")


memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1",
)

"""
### Create an LLM class instance

You can use any of the [LLMs compatible with LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/modules/).
You may need to enable Vertex AI API to use `Vertex`.
"""
logger.info("### Create an LLM class instance")


llm = Vertex(model="gemini-1.5-flash-002", project=PROJECT_ID)

"""
### Use the AlloyDBChatStore without a storage context

#### Create and use the Chat Engine
"""
logger.info("### Use the AlloyDBChatStore without a storage context")


chat_engine = SimpleChatEngine(memory=memory, llm=llm, prefix_messages=[])

response = chat_engine.chat("Hello.")

logger.debug(response)

"""
### Use the AlloyDBChatStore with a storage context

#### Create an AlloyDBVectorStore instance

Find a detailed guide on how to use the `AlloyDBVectorStore` [here](https://github.com/googleapis/llama-index-alloydb-pg-python/blob/main/samples/llama_index_vector_store.ipynb).

You can also use the `AlloyDBDocumentStore` and `AlloyDBIndexStore` to persist documents and index metadata. For a detailed python notebook on this, see [LlamaIndex Doc Store Guide](https://github.com/googleapis/llama-index-alloydb-pg-python/blob/main/samples/llama_index_doc_store.ipynb)
"""
logger.info("### Use the AlloyDBChatStore with a storage context")


await engine.ainit_vector_store_table(
    table_name=VECTOR_STORE_TABLE_NAME,
    vector_size=768,  # Vector size for VertexAI model(textembedding-gecko@latest)
)

async def async_func_7():
    vector_store = await AlloyDBVectorStore.create(
        engine=engine,
        table_name=VECTOR_STORE_TABLE_NAME,
    )
    return vector_store
vector_store = asyncio.run(async_func_7())
logger.success(format_json(vector_store))

"""
#### Create an embedding class instance

You can use any [Llama Index embeddings model](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/).
You may need to enable Vertex AI API to use `VertexTextEmbeddings`. We recommend setting the embedding model's version for production, learn more about the [Text embeddings models](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embeddings).
"""
logger.info("#### Create an embedding class instance")

# !gcloud services enable aiplatform.googleapis.com


credentials, project_id = google.auth.default()
Settings.embed_model = VertexTextEmbedding(
    model_name="textembedding-gecko@003",
    project=PROJECT_ID,
    credentials=credentials,
)

Settings.llm = Vertex(model="gemini-1.5-flash-002", project=PROJECT_ID)

"""
#### Download and load sample data
"""
logger.info("#### Download and load sample data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
logger.debug("Document ID:", documents[0].doc_id)

"""
#### Create a VectorStoreIndex with a storage context
"""
logger.info("#### Create a VectorStoreIndex with a storage context")


storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)

"""
#### Create and use the Chat Engine
"""
logger.info("#### Create and use the Chat Engine")

chat_engine = index.as_chat_engine(llm=llm, chat_mode="context", memory=memory)
response = chat_engine.chat("What did the author do?")

logger.info("\n\n[DONE]", bright=True)
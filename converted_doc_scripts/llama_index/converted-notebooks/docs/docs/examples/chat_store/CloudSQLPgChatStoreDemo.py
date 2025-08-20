import asyncio
from jet.transformers.formatters import format_json
from google.colab import auth
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.vertex import Vertex
from llama_index_cloud_sql_pg import PostgresChatStore
from llama_index_cloud_sql_pg import PostgresEngine
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
# Google Cloud SQL for PostgreSQL - `PostgresChatStore`

> [Cloud SQL](https://cloud.google.com/sql) is a fully managed relational database service that offers high performance, seamless integration, and impressive scalability. It offers MySQL, PostgreSQL, and SQL Server database engines. Extend your database application to build AI-powered experiences leveraging Cloud SQL's LlamaIndex integrations.

This notebook goes over how to use `Cloud SQL for PostgreSQL` to store chat history with `PostgresChatStore` class.

Learn more about the package on [GitHub](https://github.com/googleapis/llama-index-cloud-sql-pg-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/llama-index-cloud-sql-pg-python/blob/main/samples/llama_index_chat_store.ipynb)

## Before you begin

To run this notebook, you will need to do the following:

 * [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
 * [Enable the Cloud SQL Admin API.](https://console.cloud.google.com/flows/enableapi?apiid=sqladmin.googleapis.com)
 * [Create a Cloud SQL instance.](https://cloud.google.com/sql/docs/postgres/connect-instance-auth-proxy#create-instance)
 * [Create a Cloud SQL database.](https://cloud.google.com/sql/docs/postgres/create-manage-databases)
 * [Add a User to the database.](https://cloud.google.com/sql/docs/postgres/create-manage-users)

### 🦙 Library Installation
Install the integration library, `llama-index-cloud-sql-pg`, and the library for the embedding service, `llama-index-embeddings-vertex`.
"""
logger.info("# Google Cloud SQL for PostgreSQL - `PostgresChatStore`")

# %pip install --upgrade --quiet llama-index-cloud-sql-pg llama-index-llms-vertex llama-index

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

### Set Cloud SQL database values
Find your database values, in the [Cloud SQL Instances page](https://console.cloud.google.com/sql?_ga=2.223735448.2062268965.1707700487-2088871159.1707257687).
"""
logger.info("## Basic Usage")

REGION = "us-central1"  # @param {type: "string"}
INSTANCE = "my-primary"  # @param {type: "string"}
DATABASE = "my-database"  # @param {type: "string"}
TABLE_NAME = "chat_store"  # @param {type: "string"}
USER = "postgres"  # @param {type: "string"}
PASSWORD = "my-password"  # @param {type: "string"}

"""
### PostgresEngine Connection Pool

One of the requirements and arguments to establish Cloud SQL as a chat store is a `PostgresEngine` object. The `PostgresEngine`  configures a connection pool to your Cloud SQL database, enabling successful connections from your application and following industry best practices.

To create a `PostgresEngine` using `PostgresEngine.from_instance()` you need to provide only 4 things:

1. `project_id` : Project ID of the Google Cloud Project where the Cloud SQL instance is located.
1. `region` : Region where the Cloud SQL instance is located.
1. `instance` : The name of the Cloud SQL instance.
1. `database` : The name of the database to connect to on the Cloud SQL instance.

By default, [IAM database authentication](https://cloud.google.com/sql/docs/postgres/iam-authentication#iam-db-auth) will be used as the method of database authentication. This library uses the IAM principal belonging to the [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials) sourced from the envionment.

For more informatin on IAM database authentication please see:

* [Configure an instance for IAM database authentication](https://cloud.google.com/sql/docs/postgres/create-edit-iam-instances)
* [Manage users with IAM database authentication](https://cloud.google.com/sql/docs/postgres/add-manage-iam-users)

Optionally, [built-in database authentication](https://cloud.google.com/sql/docs/postgres/built-in-authentication) using a username and password to access the Cloud SQL database can also be used. Just provide the optional `user` and `password` arguments to `PostgresEngine.from_instance()`:

* `user` : Database user to use for built-in database authentication and login
* `password` : Database password to use for built-in database authentication and login.

**Note:** This tutorial demonstrates the async interface. All async methods have corresponding sync methods.
"""
logger.info("### PostgresEngine Connection Pool")


async def async_func_2():
    engine = await PostgresEngine.afrom_instance(
        project_id=PROJECT_ID,
        region=REGION,
        instance=INSTANCE,
        database=DATABASE,
        user=USER,
        password=PASSWORD,
    )
    return engine
engine = asyncio.run(async_func_2())
logger.success(format_json(engine))

"""
### Initialize a table
The `PostgresChatStore` class requires a database table. The `PostgresEngine` engine has a helper method `ainit_chat_store_table()` that can be used to create a table with the proper schema for you.
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
### Initialize a default PostgresChatStore
"""
logger.info("### Initialize a default PostgresChatStore")


async def async_func_2():
    chat_store = await PostgresChatStore.create(
        engine=engine,
        table_name=TABLE_NAME,
    )
    return chat_store
chat_store = asyncio.run(async_func_2())
logger.success(format_json(chat_store))

"""
### Create a ChatMemoryBuffer
The `ChatMemoryBuffer` stores a history of recent chat messages, enabling the LLM to access relevant context from prior interactions.

By passing our chat store into the `ChatMemoryBuffer`, it can automatically retrieve and update messages associated with a specific session ID or `chat_store_key`.
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
### Use the PostgresChatStore without a storage context

#### Create a Simple Chat Engine
"""
logger.info("### Use the PostgresChatStore without a storage context")


chat_engine = SimpleChatEngine(memory=memory, llm=llm, prefix_messages=[])

response = chat_engine.chat("Hello")

logger.debug(response)

"""
### Use the PostgresChatStore with a storage context

#### Create a LlamaIndex `Index`

An `Index` is allows us to quickly retrieve relevant context for a user query.
They are used to build `QueryEngines` and `ChatEngines`.
For a list of indexes that can be built in LlamaIndex, see [Index Guide](https://docs.llamaindex.ai/en/stable/module_guides/indexing/index_guide/).

A `VectorStoreIndex`, can be built using the `PostgresVectorStore`. See the detailed guide on how to use the `PostgresVectorStore` to build an index [here](https://github.com/googleapis/llama-index-cloud-sql-pg-python/blob/main/samples/llama_index_vector_store.ipynb).

You can also use the `PostgresDocumentStore` and `PostgresIndexStore` to persist documents and index metadata.
These modules can be used to build other `Indexes`.
For a detailed python notebook on this, see [LlamaIndex Doc Store Guide](https://github.com/googleapis/llama-index-cloud-sql-pg-python/blob/main/samples/llama_index_doc_store.ipynb).

#### Create and use the Chat Engine
"""
logger.info("### Use the PostgresChatStore with a storage context")

chat_engine = index.as_chat_engine(llm=llm, chat_mode="context", memory=memory)  # type: ignore
response = chat_engine.chat("What did the author do?")

logger.info("\n\n[DONE]", bright=True)
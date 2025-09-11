from google.colab import auth
from jet.logger import logger
from langchain_google_cloud_sql_mysql import Column
from langchain_google_cloud_sql_mysql import MySQLEngine
from langchain_google_cloud_sql_mysql import MySQLVectorStore
from langchain_google_cloud_sql_mysql import VectorIndex
from langchain_google_vertexai import VertexAIEmbeddings
import os
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
# Google Cloud SQL for MySQL

> [Cloud SQL](https://cloud.google.com/sql) is a fully managed relational database service that offers high performance, seamless integration, and impressive scalability. It offers PostgreSQL, MySQL, and SQL Server database engines. Extend your database application to build AI-powered experiences leveraging Cloud SQL's LangChain integrations.

This notebook goes over how to use `Cloud SQL for MySQL` to store vector embeddings with the `MySQLVectorStore` class.

Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-cloud-sql-mysql-python/blob/main/docs/vector_store.ipynb)

## Before you begin

To run this notebook, you will need to do the following:

 * [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
 * [Enable the Cloud SQL Admin API.](https://console.cloud.google.com/flows/enableapi?apiid=sqladmin.googleapis.com)
 * [Create a Cloud SQL instance.](https://cloud.google.com/sql/docs/mysql/connect-instance-auth-proxy#create-instance) (version must be >= **8.0.36** with **cloudsql_vector** database flag configured to "On")
 * [Create a Cloud SQL database.](https://cloud.google.com/sql/docs/mysql/create-manage-databases)
 * [Add a User to the database.](https://cloud.google.com/sql/docs/mysql/create-manage-users)

### 🦜🔗 Library Installation
Install the integration library, `langchain-google-cloud-sql-mysql`, and the library for the embedding service, `langchain-google-vertexai`.
"""
logger.info("# Google Cloud SQL for MySQL")

# %pip install --upgrade --quiet langchain-google-cloud-sql-mysql langchain-google-vertexai

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

**Note:** MySQL vector support is only available on MySQL instances with version **>= 8.0.36**.

For existing instances, you may need to perform a [self-service maintenance update](https://cloud.google.com/sql/docs/mysql/self-service-maintenance) to update your maintenance version to **MYSQL_8_0_36.R20240401.03_00** or greater. Once updated, [configure your database flags](https://cloud.google.com/sql/docs/mysql/flags) to have the new **cloudsql_vector** flag to "On".
"""
logger.info("## Basic Usage")

REGION = "us-central1"  # @param {type: "string"}
INSTANCE = "my-mysql-instance"  # @param {type: "string"}
DATABASE = "my-database"  # @param {type: "string"}
TABLE_NAME = "vector_store"  # @param {type: "string"}

"""
### MySQLEngine Connection Pool

One of the requirements and arguments to establish Cloud SQL as a vector store is a `MySQLEngine` object. The `MySQLEngine` configures a connection pool to your Cloud SQL database, enabling successful connections from your application and following industry best practices.

To create a `MySQLEngine` using `MySQLEngine.from_instance()` you need to provide only 4 things:

1. `project_id` : Project ID of the Google Cloud Project where the Cloud SQL instance is located.
1. `region` : Region where the Cloud SQL instance is located.
1. `instance` : The name of the Cloud SQL instance.
1. `database` : The name of the database to connect to on the Cloud SQL instance.

By default, [IAM database authentication](https://cloud.google.com/sql/docs/mysql/iam-authentication#iam-db-auth) will be used as the method of database authentication. This library uses the IAM principal belonging to the [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials) sourced from the envionment.

For more informatin on IAM database authentication please see:

* [Configure an instance for IAM database authentication](https://cloud.google.com/sql/docs/mysql/create-edit-iam-instances)
* [Manage users with IAM database authentication](https://cloud.google.com/sql/docs/mysql/add-manage-iam-users)

Optionally, [built-in database authentication](https://cloud.google.com/sql/docs/mysql/built-in-authentication) using a username and password to access the Cloud SQL database can also be used. Just provide the optional `user` and `password` arguments to `MySQLEngine.from_instance()`:

* `user` : Database user to use for built-in database authentication and login
* `password` : Database password to use for built-in database authentication and login.
"""
logger.info("### MySQLEngine Connection Pool")


engine = MySQLEngine.from_instance(
    project_id=PROJECT_ID, region=REGION, instance=INSTANCE, database=DATABASE
)

"""
### Initialize a table
The `MySQLVectorStore` class requires a database table. The `MySQLEngine` class has a helper method `init_vectorstore_table()` that can be used to create a table with the proper schema for you.
"""
logger.info("### Initialize a table")

engine.init_vectorstore_table(
    table_name=TABLE_NAME,
    vector_size=768,  # Vector size for VertexAI model(textembedding-gecko@latest)
)

"""
### Create an embedding class instance

You can use any [LangChain embeddings model](/docs/integrations/text_embedding/).
You may need to enable the Vertex AI API to use `VertexAIEmbeddings`.

We recommend pinning the embedding model's version for production, learn more about the [Text embeddings models](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embeddings).
"""
logger.info("### Create an embedding class instance")

# !gcloud services enable aiplatform.googleapis.com


embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=PROJECT_ID
)

"""
### Initialize a default MySQLVectorStore

To initialize a `MySQLVectorStore` class you need to provide only 3 things:

1. `engine` - An instance of a `MySQLEngine` engine.
1. `embedding_service` - An instance of a LangChain embedding model.
1. `table_name` : The name of the table within the Cloud SQL database to use as the vector store.
"""
logger.info("### Initialize a default MySQLVectorStore")


store = MySQLVectorStore(
    engine=engine,
    embedding_service=embedding,
    table_name=TABLE_NAME,
)

"""
### Add texts
"""
logger.info("### Add texts")


all_texts = ["Apples and oranges", "Cars and airplanes", "Pineapple", "Train", "Banana"]
metadatas = [{"len": len(t)} for t in all_texts]
ids = [str(uuid.uuid4()) for _ in all_texts]

store.add_texts(all_texts, metadatas=metadatas, ids=ids)

"""
### Delete texts

Delete vectors from the vector store by ID.
"""
logger.info("### Delete texts")

store.delete([ids[1]])

"""
### Search for documents
"""
logger.info("### Search for documents")

query = "I'd like a fruit."
docs = store.similarity_search(query)
logger.debug(docs[0].page_content)

"""
### Search for documents by vector

It is also possible to do a search for documents similar to a given embedding vector using `similarity_search_by_vector` which accepts an embedding vector as a parameter instead of a string.
"""
logger.info("### Search for documents by vector")

query_vector = embedding.embed_query(query)
docs = store.similarity_search_by_vector(query_vector, k=2)
logger.debug(docs)

"""
### Add an index
Speed up vector search queries by applying a vector index. Learn more about [MySQL vector indexes](https://github.com/googleapis/langchain-google-cloud-sql-mysql-python/blob/main/src/langchain_google_cloud_sql_mysql/indexes.py).

**Note:** For IAM database authentication (default usage), the IAM database user will need to be granted the following permissions by a privileged database user for full control of vector indexes.

```
GRANT EXECUTE ON PROCEDURE mysql.create_vector_index TO '<IAM_DB_USER>'@'%';
GRANT EXECUTE ON PROCEDURE mysql.alter_vector_index TO '<IAM_DB_USER>'@'%';
GRANT EXECUTE ON PROCEDURE mysql.drop_vector_index TO '<IAM_DB_USER>'@'%';
GRANT SELECT ON mysql.vector_indexes TO '<IAM_DB_USER>'@'%';
```
"""
logger.info("### Add an index")


store.apply_vector_index(VectorIndex())

"""
### Remove an index
"""
logger.info("### Remove an index")

store.drop_vector_index()

"""
## Advanced Usage

### Create a MySQLVectorStore with custom metadata

A vector store can take advantage of relational data to filter similarity searches.

Create a table and `MySQLVectorStore` instance with custom metadata columns.
"""
logger.info("## Advanced Usage")


CUSTOM_TABLE_NAME = "vector_store_custom"

engine.init_vectorstore_table(
    table_name=CUSTOM_TABLE_NAME,
    vector_size=768,  # VertexAI model: textembedding-gecko@latest
    metadata_columns=[Column("len", "INTEGER")],
)


custom_store = MySQLVectorStore(
    engine=engine,
    embedding_service=embedding,
    table_name=CUSTOM_TABLE_NAME,
    metadata_columns=["len"],
)

"""
### Search for documents with metadata filter

It can be helpful to narrow down the documents before working with them.

For example, documents can be filtered on metadata using the `filter` argument.
"""
logger.info("### Search for documents with metadata filter")


all_texts = ["Apples and oranges", "Cars and airplanes", "Pineapple", "Train", "Banana"]
metadatas = [{"len": len(t)} for t in all_texts]
ids = [str(uuid.uuid4()) for _ in all_texts]
custom_store.add_texts(all_texts, metadatas=metadatas, ids=ids)

query_vector = embedding.embed_query("I'd like a fruit.")
docs = custom_store.similarity_search_by_vector(query_vector, filter="len >= 6")

logger.debug(docs)

logger.info("\n\n[DONE]", bright=True)
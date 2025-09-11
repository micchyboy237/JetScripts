from jet.transformers.formatters import format_json
from google.colab import auth
from jet.logger import logger
from langchain_google_cloud_sql_pg import Column
from langchain_google_cloud_sql_pg import PostgresEngine
from langchain_google_cloud_sql_pg import PostgresVectorStore
from langchain_google_cloud_sql_pg.indexes import IVFFlatIndex
from langchain_google_vertexai import VertexAIEmbeddings
import os
import shutil
import uuid

async def main():
    
    
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
    # Google Cloud SQL for PostgreSQL
    
    > [Cloud SQL](https://cloud.google.com/sql) is a fully managed relational database service that offers high performance, seamless integration, and impressive scalability. It offers PostgreSQL, PostgreSQL, and SQL Server database engines. Extend your database application to build AI-powered experiences leveraging Cloud SQL's Langchain integrations.
    
    This notebook goes over how to use `Cloud SQL for PostgreSQL` to store vector embeddings with the `PostgresVectorStore` class.
    
    Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-cloud-sql-pg-python/).
    
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-cloud-sql-pg-python/blob/main/docs/vector_store.ipynb)
    
    ## Before you begin
    
    To run this notebook, you will need to do the following:
    
     * [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
     * [Enable the Cloud SQL Admin API.](https://console.cloud.google.com/flows/enableapi?apiid=sqladmin.googleapis.com)
     * [Create a Cloud SQL instance.](https://cloud.google.com/sql/docs/postgres/connect-instance-auth-proxy#create-instance)
     * [Create a Cloud SQL database.](https://cloud.google.com/sql/docs/postgres/create-manage-databases)
     * [Add a User to the database.](https://cloud.google.com/sql/docs/postgres/create-manage-users)
    
    ### 🦜🔗 Library Installation
    Install the integration library, `langchain-google-cloud-sql-pg`, and the library for the embedding service, `langchain-google-vertexai`.
    """
    logger.info("# Google Cloud SQL for PostgreSQL")
    
    # %pip install --upgrade --quiet  langchain-google-cloud-sql-pg langchain-google-vertexai
    
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
    INSTANCE = "my-pg-instance"  # @param {type: "string"}
    DATABASE = "my-database"  # @param {type: "string"}
    TABLE_NAME = "vector_store"  # @param {type: "string"}
    
    """
    ### PostgresEngine Connection Pool
    
    One of the requirements and arguments to establish Cloud SQL as a vector store is a `PostgresEngine` object. The `PostgresEngine`  configures a connection pool to your Cloud SQL database, enabling successful connections from your application and following industry best practices.
    
    To create a `PostgresEngine` using `PostgresEngine.from_instance()` you need to provide only 4 things:
    
    1.   `project_id` : Project ID of the Google Cloud Project where the Cloud SQL instance is located.
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
    
    "**Note**: This tutorial demonstrates the async interface. All async methods have corresponding sync methods."
    """
    logger.info("### PostgresEngine Connection Pool")
    
    
    engine = await PostgresEngine.afrom_instance(
            project_id=PROJECT_ID, region=REGION, instance=INSTANCE, database=DATABASE
        )
    logger.success(format_json(engine))
    
    """
    ### Initialize a table
    The `PostgresVectorStore` class requires a database table. The `PostgresEngine` engine has a helper method `init_vectorstore_table()` that can be used to create a table with the proper schema for you.
    """
    logger.info("### Initialize a table")
    
    
    await engine.ainit_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=768,  # Vector size for VertexAI model(textembedding-gecko@latest)
    )
    
    """
    ### Create an embedding class instance
    
    You can use any [LangChain embeddings model](/docs/integrations/text_embedding/).
    You may need to enable Vertex AI API to use `VertexAIEmbeddings`. We recommend setting the embedding model's version for production, learn more about the [Text embeddings models](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embeddings).
    """
    logger.info("### Create an embedding class instance")
    
    # !gcloud services enable aiplatform.googleapis.com
    
    
    embedding = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest", project=PROJECT_ID
    )
    
    """
    ### Initialize a default PostgresVectorStore
    """
    logger.info("### Initialize a default PostgresVectorStore")
    
    
    store = await PostgresVectorStore.create(  # Use .create() to initialize an async vector store
            engine=engine,
            table_name=TABLE_NAME,
            embedding_service=embedding,
        )
    logger.success(format_json(store))
    
    """
    ### Add texts
    """
    logger.info("### Add texts")
    
    
    all_texts = ["Apples and oranges", "Cars and airplanes", "Pineapple", "Train", "Banana"]
    metadatas = [{"len": len(t)} for t in all_texts]
    ids = [str(uuid.uuid4()) for _ in all_texts]
    
    await store.aadd_texts(all_texts, metadatas=metadatas, ids=ids)
    
    """
    ### Delete texts
    """
    logger.info("### Delete texts")
    
    await store.adelete([ids[1]])
    
    """
    ### Search for documents
    """
    logger.info("### Search for documents")
    
    query = "I'd like a fruit."
    docs = await store.asimilarity_search(query)
    logger.success(format_json(docs))
    logger.debug(docs)
    
    """
    ### Search for documents by vector
    """
    logger.info("### Search for documents by vector")
    
    query_vector = embedding.embed_query(query)
    docs = await store.asimilarity_search_by_vector(query_vector, k=2)
    logger.success(format_json(docs))
    logger.debug(docs)
    
    """
    ## Add a Index
    Speed up vector search queries by applying a vector index. Learn more about [vector indexes](https://cloud.google.com/blog/products/databases/faster-similarity-search-performance-with-pgvector-indexes).
    """
    logger.info("## Add a Index")
    
    
    index = IVFFlatIndex()
    await store.aapply_vector_index(index)
    
    """
    ### Re-index
    """
    logger.info("### Re-index")
    
    await store.areindex()  # Re-index using default index name
    
    """
    ### Remove an index
    """
    logger.info("### Remove an index")
    
    await store.aadrop_vector_index()  # Delete index using default name
    
    """
    ## Create a custom Vector Store
    A Vector Store can take advantage of relational data to filter similarity searches.
    
    Create a table with custom metadata columns.
    """
    logger.info("## Create a custom Vector Store")
    
    
    TABLE_NAME = "vectorstore_custom"
    
    await engine.ainit_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=768,  # VertexAI model: textembedding-gecko@latest
        metadata_columns=[Column("len", "INTEGER")],
    )
    
    
    custom_store = await PostgresVectorStore.create(
            engine=engine,
            table_name=TABLE_NAME,
            embedding_service=embedding,
            metadata_columns=["len"],
        )
    logger.success(format_json(custom_store))
    
    """
    ### Search for documents with metadata filter
    """
    logger.info("### Search for documents with metadata filter")
    
    
    all_texts = ["Apples and oranges", "Cars and airplanes", "Pineapple", "Train", "Banana"]
    metadatas = [{"len": len(t)} for t in all_texts]
    ids = [str(uuid.uuid4()) for _ in all_texts]
    await store.aadd_texts(all_texts, metadatas=metadatas, ids=ids)
    
    docs = await custom_store.asimilarity_search_by_vector(query_vector, filter="len >= 6")
    logger.success(format_json(docs))
    
    logger.debug(docs)
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())
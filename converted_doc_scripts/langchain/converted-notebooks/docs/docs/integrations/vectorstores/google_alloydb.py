from jet.transformers.formatters import format_json
from google.colab import auth
from jet.logger import logger
from langchain_google_alloydb_pg import AlloyDBEngine
from langchain_google_alloydb_pg import AlloyDBVectorStore
from langchain_google_alloydb_pg import Column
from langchain_google_alloydb_pg.indexes import IVFFlatIndex
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
    # Google AlloyDB for PostgreSQL
    
    > [AlloyDB](https://cloud.google.com/alloydb) is a fully managed relational database service that offers high performance, seamless integration, and impressive scalability. AlloyDB is 100% compatible with PostgreSQL. Extend your database application to build AI-powered experiences leveraging AlloyDB's Langchain integrations.
    
    This notebook goes over how to use `AlloyDB for PostgreSQL` to store vector embeddings with the `AlloyDBVectorStore` class.
    
    Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-alloydb-pg-python/).
    
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-alloydb-pg-python/blob/main/docs/vector_store.ipynb)
    
    ## Before you begin
    
    To run this notebook, you will need to do the following:
    
     * [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
     * [Enable the AlloyDB API](https://console.cloud.google.com/flows/enableapi?apiid=alloydb.googleapis.com)
     * [Create a AlloyDB cluster and instance.](https://cloud.google.com/alloydb/docs/cluster-create)
     * [Create a AlloyDB database.](https://cloud.google.com/alloydb/docs/quickstart/create-and-connect)
     * [Add a User to the database.](https://cloud.google.com/alloydb/docs/database-users/about)
    
    ### 🦜🔗 Library Installation
    Install the integration library, `langchain-google-alloydb-pg`, and the library for the embedding service, `langchain-google-vertexai`.
    """
    logger.info("# Google AlloyDB for PostgreSQL")
    
    # %pip install --upgrade --quiet  langchain-google-alloydb-pg langchain-google-vertexai
    
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
    TABLE_NAME = "vector_store"  # @param {type: "string"}
    
    """
    ### AlloyDBEngine Connection Pool
    
    One of the requirements and arguments to establish AlloyDB as a vector store is a `AlloyDBEngine` object. The `AlloyDBEngine`  configures a connection pool to your AlloyDB database, enabling successful connections from your application and following industry best practices.
    
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
    
    
    engine = await AlloyDBEngine.afrom_instance(
            project_id=PROJECT_ID,
            region=REGION,
            cluster=CLUSTER,
            instance=INSTANCE,
            database=DATABASE,
        )
    logger.success(format_json(engine))
    
    """
    ### Initialize a table
    The `AlloyDBVectorStore` class requires a database table. The `AlloyDBEngine` engine has a helper method `init_vectorstore_table()` that can be used to create a table with the proper schema for you.
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
    ### Initialize a default AlloyDBVectorStore
    """
    logger.info("### Initialize a default AlloyDBVectorStore")
    
    
    store = await AlloyDBVectorStore.create(
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
    
    await store.adrop_vector_index()  # Delete index using default name
    
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
    
    
    custom_store = await AlloyDBVectorStore.create(
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
from jet.transformers.formatters import format_json
from google.colab import auth
from jet.logger import logger
from langchain_google_alloydb_pg import AlloyDBEngine
from langchain_google_alloydb_pg import AlloyDBLoader
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
# Google AlloyDB for PostgreSQL

> [AlloyDB](https://cloud.google.com/alloydb) is a fully managed relational database service that offers high performance, seamless integration, and impressive scalability. AlloyDB is 100% compatible with PostgreSQL. Extend your database application to build AI-powered experiences leveraging AlloyDB's Langchain integrations.

This notebook goes over how to use `AlloyDB for PostgreSQL` to load Documents with the `AlloyDBLoader` class.

Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-alloydb-pg-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-alloydb-pg-python/blob/main/docs/document_loader.ipynb)

## Before you begin

To run this notebook, you will need to do the following:

 * [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
 * [Enable the AlloyDB API](https://console.cloud.google.com/flows/enableapi?apiid=alloydb.googleapis.com)
 * [Create a AlloyDB cluster and instance.](https://cloud.google.com/alloydb/docs/cluster-create)
 * [Create a AlloyDB database.](https://cloud.google.com/alloydb/docs/quickstart/create-and-connect)
 * [Add a User to the database.](https://cloud.google.com/alloydb/docs/database-users/about)

### 🦜🔗 Library Installation
Install the integration library, `langchain-google-alloydb-pg`.
"""
logger.info("# Google AlloyDB for PostgreSQL")

# %pip install --upgrade --quiet  langchain-google-alloydb-pg

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

PROJECT_ID = "gcp_project_id"  # @param {type:"string"}

# ! gcloud config set project {PROJECT_ID}

"""
## Basic Usage

### Set AlloyDB database variables
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

**Note**: This tutorial demonstrates the async interface. All async methods have corresponding sync methods.
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
### Create AlloyDBLoader
"""
logger.info("### Create AlloyDBLoader")


loader = await AlloyDBLoader.create(engine, table_name=TABLE_NAME)
logger.success(format_json(loader))

"""
### Load Documents via default table
The loader returns a list of Documents from the table using the first column as page_content and all other columns as metadata. The default table will have the first column as
page_content and the second column as metadata (JSON). Each row becomes a document.
"""
logger.info("### Load Documents via default table")

docs = await loader.aload()
logger.success(format_json(docs))
logger.debug(docs)

"""
### Load documents via custom table/metadata or custom page content columns
"""
logger.info("### Load documents via custom table/metadata or custom page content columns")

loader = await AlloyDBLoader.create(
        engine,
        table_name=TABLE_NAME,
        content_columns=["product_name"],  # Optional
        metadata_columns=["id"],  # Optional
    )
logger.success(format_json(loader))
docs = await loader.aload()
logger.success(format_json(docs))
logger.debug(docs)

"""
### Set page content format
The loader returns a list of Documents, with one document per row, with page content in specified string format, i.e. text (space separated concatenation), JSON, YAML, CSV, etc. JSON and YAML formats include headers, while text and CSV do not include field headers.
"""
logger.info("### Set page content format")

loader = AlloyDBLoader.create(
    engine,
    table_name="products",
    content_columns=["product_name", "description"],
    format="YAML",
)
docs = await loader.aload()
logger.success(format_json(docs))
logger.debug(docs)

logger.info("\n\n[DONE]", bright=True)
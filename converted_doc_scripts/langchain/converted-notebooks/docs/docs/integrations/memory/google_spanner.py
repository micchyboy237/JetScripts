from google.cloud import spanner
from google.colab import auth
from jet.logger import logger
from langchain_google_spanner import (
SpannerChatMessageHistory,
)
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
# Google Spanner
> [Google Cloud Spanner](https://cloud.google.com/spanner) is a highly scalable database that combines unlimited scalability with relational semantics, such as secondary indexes, strong consistency, schemas, and SQL providing 99.999% availability in one easy solution.

This notebook goes over how to use `Spanner` to store chat message history with the `SpannerChatMessageHistory` class.
Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-spanner-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-spanner-python/blob/main/samples/chat_message_history.ipynb)

## Before You Begin

To run this notebook, you will need to do the following:

 * [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
 * [Enable the Cloud Spanner API](https://console.cloud.google.com/flows/enableapi?apiid=spanner.googleapis.com)
 * [Create a Spanner instance](https://cloud.google.com/spanner/docs/create-manage-instances)
 * [Create a Spanner database](https://cloud.google.com/spanner/docs/create-manage-databases)

### 🦜🔗 Library Installation
The integration lives in its own `langchain-google-spanner` package, so we need to install it.
"""
logger.info("# Google Spanner")

# %pip install --upgrade --quiet langchain-google-spanner

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
### 💡 API Enablement
The `langchain-google-spanner` package requires that you [enable the Spanner API](https://console.cloud.google.com/flows/enableapi?apiid=spanner.googleapis.com) in your Google Cloud Project.
"""
logger.info("### 💡 API Enablement")

# !gcloud services enable spanner.googleapis.com

"""
## Basic Usage

### Set Spanner database values
Find your database values, in the [Spanner Instances page](https://console.cloud.google.com/spanner).
"""
logger.info("## Basic Usage")

INSTANCE = "my-instance"  # @param {type: "string"}
DATABASE = "my-database"  # @param {type: "string"}
TABLE_NAME = "message_store"  # @param {type: "string"}

"""
### Initialize a table
The `SpannerChatMessageHistory` class requires a database table with a specific schema in order to store the chat message history.

The helper method `init_chat_history_table()` that can be used to create a table with the proper schema for you.
"""
logger.info("### Initialize a table")


SpannerChatMessageHistory.init_chat_history_table(table_name=TABLE_NAME)

"""
### SpannerChatMessageHistory

To initialize the `SpannerChatMessageHistory` class you need to provide only 3 things:

1. `instance_id` - The name of the Spanner instance
1. `database_id` - The name of the Spanner database
1. `session_id` - A unique identifier string that specifies an id for the session.
1. `table_name` - The name of the table within the database to store the chat message history.
"""
logger.info("### SpannerChatMessageHistory")

message_history = SpannerChatMessageHistory(
    instance_id=INSTANCE,
    database_id=DATABASE,
    table_name=TABLE_NAME,
    session_id="user-session-id",
)

message_history.add_user_message("hi!")
message_history.add_ai_message("whats up?")

message_history.messages

"""
## Custom client
The client created by default is the default client. To use a non-default, a [custom client](https://cloud.google.com/spanner/docs/samples/spanner-create-client-with-query-options#spanner_create_client_with_query_options-python) can be passed to the constructor.
"""
logger.info("## Custom client")


custom_client_message_history = SpannerChatMessageHistory(
    instance_id="my-instance",
    database_id="my-database",
    client=spanner.Client(...),
)

"""
## Cleaning up

When the history of a specific session is obsolete and can be deleted, it can be done the following way.
Note: Once deleted, the data is no longer stored in Cloud Spanner and is gone forever.
"""
logger.info("## Cleaning up")

message_history = SpannerChatMessageHistory(
    instance_id=INSTANCE,
    database_id=DATABASE,
    table_name=TABLE_NAME,
    session_id="user-session-id",
)

message_history.clear()

logger.info("\n\n[DONE]", bright=True)
from google.cloud import bigtable
from google.colab import auth
from jet.logger import logger
from langchain_google_bigtable import BigtableChatMessageHistory
from langchain_google_bigtable import create_chat_history_table
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
# Google Bigtable

> [Google Cloud Bigtable](https://cloud.google.com/bigtable) is a key-value and wide-column store, ideal for fast access to structured, semi-structured, or unstructured data. Extend your database application to build AI-powered experiences leveraging Bigtable's Langchain integrations.

This notebook goes over how to use [Google Cloud Bigtable](https://cloud.google.com/bigtable) to store chat message history with the `BigtableChatMessageHistory` class.

Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-bigtable-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-bigtable-python/blob/main/docs/chat_message_history.ipynb)

## Before You Begin

To run this notebook, you will need to do the following:

* [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
* [Enable the Bigtable API](https://console.cloud.google.com/flows/enableapi?apiid=bigtable.googleapis.com)
* [Create a Bigtable instance](https://cloud.google.com/bigtable/docs/creating-instance)
* [Create a Bigtable table](https://cloud.google.com/bigtable/docs/managing-tables)
* [Create Bigtable access credentials](https://developers.google.com/workspace/guides/create-credentials)

### 🦜🔗 Library Installation

The integration lives in its own `langchain-google-bigtable` package, so we need to install it.
"""
logger.info("# Google Bigtable")

# %pip install -upgrade --quiet langchain-google-bigtable

"""
**Colab only**: Uncomment the following cell to restart the kernel or use the button to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.
"""



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
### 🔐 Authentication

Authenticate to Google Cloud as the IAM user logged into this notebook in order to access your Google Cloud Project.

- If you are using Colab to run this notebook, use the cell below and continue.
- If you are using Vertex AI Workbench, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).
"""
logger.info("### 🔐 Authentication")


auth.authenticate_user()

"""
## Basic Usage

### Initialize Bigtable schema

The schema for BigtableChatMessageHistory requires the instance and table to exist, and have a column family called `langchain`.
"""
logger.info("## Basic Usage")

INSTANCE_ID = "my_instance"  # @param {type:"string"}
TABLE_ID = "my_table"  # @param {type:"string"}

"""
If the table or the column family do not exist, you can use the following function to create them:
"""
logger.info("If the table or the column family do not exist, you can use the following function to create them:")


create_chat_history_table(
    instance_id=INSTANCE_ID,
    table_id=TABLE_ID,
)

"""
### BigtableChatMessageHistory

To initialize the `BigtableChatMessageHistory` class you need to provide only 3 things:

1. `instance_id` - The Bigtable instance to use for chat message history.
1. `table_id` : The Bigtable table to store the chat message history.
1. `session_id` - A unique identifier string that specifies an id for the session.
"""
logger.info("### BigtableChatMessageHistory")


message_history = BigtableChatMessageHistory(
    instance_id=INSTANCE_ID,
    table_id=TABLE_ID,
    session_id="user-session-id",
)

message_history.add_user_message("hi!")
message_history.add_ai_message("whats up?")

message_history.messages

"""
#### Cleaning up

When the history of a specific session is obsolete and can be deleted, it can be done the following way.

**Note:** Once deleted, the data is no longer stored in Bigtable and is gone forever.
"""
logger.info("#### Cleaning up")

message_history.clear()

"""
## Advanced Usage

### Custom client
The client created by default is the default client, using only admin=True option. To use a non-default, a [custom client](https://cloud.google.com/python/docs/reference/bigtable/latest/client#class-googlecloudbigtableclientclientprojectnone-credentialsnone-readonlyfalse-adminfalse-clientinfonone-clientoptionsnone-adminclientoptionsnone-channelnone) can be passed to the constructor.
"""
logger.info("## Advanced Usage")


client = (bigtable.Client(...),)

create_chat_history_table(
    instance_id="my-instance",
    table_id="my-table",
    client=client,
)

custom_client_message_history = BigtableChatMessageHistory(
    instance_id="my-instance",
    table_id="my-table",
    client=client,
)

logger.info("\n\n[DONE]", bright=True)
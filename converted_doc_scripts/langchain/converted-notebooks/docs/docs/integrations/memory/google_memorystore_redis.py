from google.colab import auth
from jet.logger import logger
from langchain_google_memorystore_redis import MemorystoreChatMessageHistory
import os
import redis
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
# Google Memorystore for Redis

> [Google Cloud Memorystore for Redis](https://cloud.google.com/memorystore/docs/redis/memorystore-for-redis-overview) is a fully-managed service that is powered by the Redis in-memory data store to build application caches that provide sub-millisecond data access. Extend your database application to build AI-powered experiences leveraging Memorystore for Redis's Langchain integrations.

This notebook goes over how to use [Google Cloud Memorystore for Redis](https://cloud.google.com/memorystore/docs/redis/memorystore-for-redis-overview) to store chat message history with the `MemorystoreChatMessageHistory` class.

Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-memorystore-redis-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-memorystore-redis-python/blob/main/docs/chat_message_history.ipynb)

## Before You Begin

To run this notebook, you will need to do the following:

* [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
* [Enable the Memorystore for Redis API](https://console.cloud.google.com/flows/enableapi?apiid=redis.googleapis.com)
* [Create a Memorystore for Redis instance](https://cloud.google.com/memorystore/docs/redis/create-instance-console). Ensure that the version is greater than or equal to 5.0.

After confirmed access to database in the runtime environment of this notebook, filling the following values and run the cell before running example scripts.
"""
logger.info("# Google Memorystore for Redis")

ENDPOINT = "redis://127.0.0.1:6379"  # @param {type:"string"}

"""
### 🦜🔗 Library Installation

The integration lives in its own `langchain-google-memorystore-redis` package, so we need to install it.
"""
logger.info("### 🦜🔗 Library Installation")

# %pip install -upgrade --quiet langchain-google-memorystore-redis

"""
**Colab only:** Uncomment the following cell to restart the kernel or use the button to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.
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

* If you are using Colab to run this notebook, use the cell below and continue.
* If you are using Vertex AI Workbench, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).
"""
logger.info("### 🔐 Authentication")


auth.authenticate_user()

"""
## Basic Usage

### MemorystoreChatMessageHistory

To initialize the `MemorystoreMessageHistory` class you need to provide only 2 things:

1. `redis_client` - An instance of a Memorystore Redis.
1. `session_id` - Each chat message history object must have a unique session ID. If the session ID already has messages stored in Redis, they will can be retrieved.
"""
logger.info("## Basic Usage")


redis_client = redis.from_url("redis://127.0.0.1:6379")

message_history = MemorystoreChatMessageHistory(redis_client, session_id="session1")

message_history.messages

"""
#### Cleaning up

When the history of a specific session is obsolete and can be deleted, it can be done the following way.

**Note:** Once deleted, the data is no longer stored in Memorystore for Redis and is gone forever.
"""
logger.info("#### Cleaning up")

message_history.clear()

logger.info("\n\n[DONE]", bright=True)
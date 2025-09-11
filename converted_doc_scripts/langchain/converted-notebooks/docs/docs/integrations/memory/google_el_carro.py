from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_el_carro import ElCarroChatMessageHistory
from langchain_google_el_carro import ElCarroEngine
from langchain_google_vertexai import ChatVertexAI
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
# Google El Carro Oracle

> [Google Cloud El Carro Oracle](https://github.com/GoogleCloudPlatform/elcarro-oracle-operator) offers a way to run `Oracle` databases in `Kubernetes` as a portable, open source, community-driven, no vendor lock-in container orchestration system. `El Carro` provides a powerful declarative API for comprehensive and consistent configuration and deployment as well as for real-time operations and monitoring. Extend your `Oracle` database's capabilities to build AI-powered experiences by leveraging the `El Carro` Langchain integration.

This guide goes over how to use the `El Carro` Langchain integration to store chat message history with the `ElCarroChatMessageHistory` class. This integration works for any `Oracle` database, regardless of where it is running.

Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-el-carro-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-el-carro-python/blob/main/docs/chat_message_history.ipynb)

## Before You Begin

To run this notebook, you will need to do the following:

 * Complete the [Getting Started](https://github.com/googleapis/langchain-google-el-carro-python/tree/main/README.md#getting-started) section if you would like to run your Oracle database with El Carro.

### 🦜🔗 Library Installation
The integration lives in its own `langchain-google-el-carro` package, so we need to install it.
"""
logger.info("# Google El Carro Oracle")

# %pip install --upgrade --quiet langchain-google-el-carro langchain-google-vertexai langchain

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

### Set Up Oracle Database Connection
Fill out the following variable with your Oracle database connections details.
"""
logger.info("## Basic Usage")

HOST = "127.0.0.1"  # @param {type: "string"}
PORT = 3307  # @param {type: "integer"}
DATABASE = "my-database"  # @param {type: "string"}
TABLE_NAME = "message_store"  # @param {type: "string"}
USER = "my-user"  # @param {type: "string"}
PASSWORD = input("Please provide a password to be used for the database user: ")

"""
If you are using `El Carro`, you can find the hostname and port values in the
status of the `El Carro` Kubernetes instance.
Use the user password you created for your PDB.
Example

kubectl get -w instances.oracle.db.anthosapis.com -n db
NAME   DB ENGINE   VERSION   EDITION      ENDPOINT      URL                DB NAMES   BACKUP ID   READYSTATUS   READYREASON        DBREADYSTATUS   DBREADYREASON
mydb   Oracle      18c       Express      mydb-svc.db   34.71.69.25:6021                          False         CreateInProgress

### ElCarroEngine Connection Pool

`ElCarroEngine` configures a connection pool to your Oracle database, enabling successful connections from your application and following industry best practices.
"""
logger.info("### ElCarroEngine Connection Pool")


elcarro_engine = ElCarroEngine.from_instance(
    db_host=HOST,
    db_port=PORT,
    db_name=DATABASE,
    db_user=USER,
    db_password=PASSWORD,
)

"""
### Initialize a table
The `ElCarroChatMessageHistory` class requires a database table with a specific
schema in order to store the chat message history.

The `ElCarroEngine` class has a
method `init_chat_history_table()` that can be used to create a table with the
proper schema for you.
"""
logger.info("### Initialize a table")

elcarro_engine.init_chat_history_table(table_name=TABLE_NAME)

"""
### ElCarroChatMessageHistory

To initialize the `ElCarroChatMessageHistory` class you need to provide only 3
things:

1. `elcarro_engine` - An instance of an `ElCarroEngine` engine.
1. `session_id` - A unique identifier string that specifies an id for the
   session.
1. `table_name` : The name of the table within the Oracle database to store the
   chat message history.
"""
logger.info("### ElCarroChatMessageHistory")


history = ElCarroChatMessageHistory(
    elcarro_engine=elcarro_engine, session_id="test_session", table_name=TABLE_NAME
)
history.add_user_message("hi!")
history.add_ai_message("whats up?")

history.messages

"""
#### Cleaning up
When the history of a specific session is obsolete and can be deleted, it can be done the following way.

**Note:** Once deleted, the data is no longer stored in your database and is gone forever.
"""
logger.info("#### Cleaning up")

history.clear()

"""
## 🔗 Chaining

We can easily combine this message history class with [LCEL Runnables](/docs/how_to/message_history)

To do this we will use one of [Google's Vertex AI chat models](/docs/integrations/chat/google_vertex_ai_palm) which requires that you [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com) in your Google Cloud Project.
"""
logger.info("## 🔗 Chaining")

# !gcloud services enable aiplatform.googleapis.com


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatVertexAI(project=PROJECT_ID)

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: ElCarroChatMessageHistory(
        elcarro_engine,
        session_id=session_id,
        table_name=TABLE_NAME,
    ),
    input_messages_key="question",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "test_session"}}

chain_with_history.invoke({"question": "Hi! I'm bob"}, config=config)

chain_with_history.invoke({"question": "Whats my name"}, config=config)

logger.info("\n\n[DONE]", bright=True)
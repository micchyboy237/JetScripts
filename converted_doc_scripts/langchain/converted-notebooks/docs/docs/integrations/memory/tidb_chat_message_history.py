from datetime import datetime
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.chat_message_histories import TiDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
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
# TiDB

> [TiDB Cloud](https://www.pingcap.com/tidb-serverless/), is a comprehensive Database-as-a-Service (DBaaS) solution, that provides dedicated and serverless options. TiDB Serverless is now integrating a built-in vector search into the MySQL landscape. With this enhancement, you can seamlessly develop AI applications using TiDB Serverless without the need for a new database or additional technical stacks. Create a free TiDB Serverless cluster and start using the vector search feature at https://pingcap.com/ai.

This notebook introduces how to use TiDB to store chat message history.

## Setup

Firstly, we will install the following dependencies:
"""
logger.info("# TiDB")

# %pip install --upgrade --quiet langchain jet.adapters.langchain.chat_ollama langchain-community

"""
Configuring your Ollama Key
"""
logger.info("Configuring your Ollama Key")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Input your Ollama API key:")

"""
Finally, we will configure the connection to a TiDB. In this notebook, we will follow the standard connection method provided by TiDB Cloud to establish a secure and efficient database connection.
"""
logger.info("Finally, we will configure the connection to a TiDB. In this notebook, we will follow the standard connection method provided by TiDB Cloud to establish a secure and efficient database connection.")

tidb_connection_string_template = "mysql+pymysql://<USER>:<PASSWORD>@<HOST>:4000/<DB>?ssl_ca=/etc/ssl/cert.pem&ssl_verify_cert=true&ssl_verify_identity=true"
# tidb_password = getpass.getpass("Input your TiDB password:")
tidb_connection_string = tidb_connection_string_template.replace(
    "<PASSWORD>", tidb_password
)

"""
## Generating historical data

Creating a set of historical data, which will serve as the foundation for our upcoming demonstrations.
"""
logger.info("## Generating historical data")



history = TiDBChatMessageHistory(
    connection_string=tidb_connection_string,
    session_id="code_gen",
    earliest_time=datetime.utcnow(),  # Optional to set earliest_time to load messages after this time point.
)

history.add_user_message("How's our feature going?")
history.add_ai_message(
    "It's going well. We are working on testing now. It will be released in Feb."
)

history.messages

"""
## Chatting with historical data

Let’s build upon the historical data generated earlier to create a dynamic chat interaction.  

Firstly, Creating a Chat Chain with LangChain:
"""
logger.info("## Chatting with historical data")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who's good at coding. You're helping a startup build",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
chain = prompt | ChatOllama(model="llama3.2")

"""
Building a Runnable on History:
"""
logger.info("Building a Runnable on History:")


chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: TiDBChatMessageHistory(
        session_id=session_id, connection_string=tidb_connection_string
    ),
    input_messages_key="question",
    history_messages_key="history",
)

"""
Initiating the Chat:
"""
logger.info("Initiating the Chat:")

response = chain_with_history.invoke(
    {"question": "Today is Jan 1st. How many days until our feature is released?"},
    config={"configurable": {"session_id": "code_gen"}},
)
response

"""
## Checking the history data
"""
logger.info("## Checking the history data")

history.reload_cache()
history.messages

logger.info("\n\n[DONE]", bright=True)
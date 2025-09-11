from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.chat_message_histories import SQLChatMessageHistory
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
# SQLite

>[SQLite](https://en.wikipedia.org/wiki/SQLite) is a database engine written in the C programming language. It is not a standalone app; rather, it is a library that software developers embed in their apps. As such, it belongs to the family of embedded databases. It is the most widely deployed database engine, as it is used by several of the top web browsers, operating systems, mobile phones, and other embedded systems.

In this walkthrough we'll create a simple conversation chain which uses `ConversationEntityMemory` backed by a `SqliteEntityStore`.
"""
logger.info("# SQLite")



"""
## Usage

To use the storage you need to provide only 2 things:

1. Session Id - a unique identifier of the session, like user name, email, chat id etc.
2. Connection string - a string that specifies the database connection. For SQLite, that string is `slqlite:///` followed by the name of the database file.  If that file doesn't exist, it will be created.
"""
logger.info("## Usage")


chat_message_history = SQLChatMessageHistory(
    session_id="test_session_id", connection_string="sqlite:///sqlite.db"
)

chat_message_history.add_user_message("Hello")
chat_message_history.add_ai_message("Hi")

chat_message_history.messages

"""
## Chaining

We can easily combine this message history class with [LCEL Runnables](/docs/how_to/message_history)

# To do this we will want to use Ollama, so we need to install that.  We will also need to set the OPENAI_API_KEY environment variable to your Ollama key.

```bash
pip install -U langchain-ollama

# export OPENAI_API_KEY='sk-xxxxxxx'
```
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatOllama(model="llama3.2")

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///sqlite.db"
    ),
    input_messages_key="question",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "<SQL_SESSION_ID>"}}

chain_with_history.invoke({"question": "Hi! I'm bob"}, config=config)

chain_with_history.invoke({"question": "Whats my name"}, config=config)

logger.info("\n\n[DONE]", bright=True)
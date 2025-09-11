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
# SQL (SQLAlchemy)

>[Structured Query Language (SQL)](https://en.wikipedia.org/wiki/SQL) is a domain-specific language used in programming and designed for managing data held in a relational database management system (RDBMS), or for stream processing in a relational data stream management system (RDSMS). It is particularly useful in handling structured data, i.e., data incorporating relations among entities and variables.

>[SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) is an open-source `SQL` toolkit and object-relational mapper (ORM) for the Python programming language released under the MIT License.

This notebook goes over a `SQLChatMessageHistory` class that allows to store chat history in any database supported by `SQLAlchemy`.

Please note that to use it with databases other than `SQLite`, you will need to install the corresponding database driver.

## Setup

The integration lives in the `langchain-community` package, so we need to install that. We also need to install the `SQLAlchemy` package.

```bash
pip install -U langchain-community SQLAlchemy langchain-ollama
```

It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability
"""
logger.info("# SQL (SQLAlchemy)")



"""
## Usage

To use the storage you need to provide only 2 things:

1. Session Id - a unique identifier of the session, like user name, email, chat id etc.
2. Connection string - a string that specifies the database connection. It will be passed to SQLAlchemy create_engine function.
"""
logger.info("## Usage")


chat_message_history = SQLChatMessageHistory(
    session_id="test_session", connection_string="sqlite:///sqlite.db"
)

chat_message_history.add_user_message("Hello")
chat_message_history.add_ai_message("Hi")

chat_message_history.messages

"""
## Chaining

We can easily combine this message history class with [LCEL Runnables](/docs/how_to/message_history)

To do this we will want to use Ollama, so we need to install that
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

config = {"configurable": {"session_id": "<SESSION_ID>"}}

chain_with_history.invoke({"question": "Hi! I'm bob"}, config=config)

chain_with_history.invoke({"question": "Whats my name"}, config=config)

logger.info("\n\n[DONE]", bright=True)
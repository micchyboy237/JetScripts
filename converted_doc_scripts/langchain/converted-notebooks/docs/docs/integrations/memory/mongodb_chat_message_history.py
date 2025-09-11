from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
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
# MongoDB

>`MongoDB` is a source-available cross-platform document-oriented database program. Classified as a NoSQL database program, `MongoDB` uses `JSON`-like documents with optional schemas.
>
>`MongoDB` is developed by MongoDB Inc. and licensed under the Server Side Public License (SSPL). - [Wikipedia](https://en.wikipedia.org/wiki/MongoDB)

This notebook goes over how to use the `MongoDBChatMessageHistory` class to store chat message history in a Mongodb database.

## Setup

The integration lives in the `langchain-mongodb` package, so we need to install that.

```bash
pip install -U --quiet langchain-mongodb
```

It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability
"""
logger.info("# MongoDB")



"""
## Usage

To use the storage you need to provide only 2 things:

1. Session Id - a unique identifier of the session, like user name, email, chat id etc.
2. Connection string - a string that specifies the database connection. It will be passed to MongoDB create_engine function.

If you want to customize where the chat histories go, you can also pass:
1. *database_name* - name of the database to use
1. *collection_name* - collection to use within that database
"""
logger.info("## Usage")


chat_message_history = MongoDBChatMessageHistory(
    session_id="test_session",
    connection_string="mongodb://mongo_user:password123@mongo:27017",
    database_name="my_db",
    collection_name="chat_histories",
)

chat_message_history.add_user_message("Hello")
chat_message_history.add_ai_message("Hi")

chat_message_history.messages

"""
## Chaining

We can easily combine this message history class with [LCEL Runnables](/docs/how_to/message_history)

# To do this we will want to use Ollama, so we need to install that.  You will also need to set the OPENAI_API_KEY environment variable to your Ollama key.
"""
logger.info("## Chaining")



# assert os.environ["OPENAI_API_KEY"], (
#     "Set the OPENAI_API_KEY environment variable with your Ollama API key."
)

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
    lambda session_id: MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string="mongodb://mongo_user:password123@mongo:27017",
        database_name="my_db",
        collection_name="chat_histories",
    ),
    input_messages_key="question",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "<SESSION_ID>"}}

chain_with_history.invoke({"question": "Hi! I'm bob"}, config=config)

chain_with_history.invoke({"question": "Whats my name"}, config=config)

logger.info("\n\n[DONE]", bright=True)
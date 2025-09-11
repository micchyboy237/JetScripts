from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_redis import RedisChatMessageHistory
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
# Redis Chat Message History

>[Redis (Remote Dictionary Server)](https://en.wikipedia.org/wiki/Redis) is an open-source in-memory storage, used as a distributed, in-memory key–value database, cache and message broker, with optional durability. `Redis` offers low-latency reads and writes. Redis is the most popular NoSQL database, and one of the most popular databases overall.

This notebook demonstrates how to use the `RedisChatMessageHistory` class from the langchain-redis package to store and manage chat message history using Redis.

## Setup

First, we need to install the required dependencies and ensure we have a Redis instance running.
"""
logger.info("# Redis Chat Message History")

# %pip install -qU langchain-redis langchain-ollama redis

"""
Make sure you have a Redis server running. You can start one using Docker with the following command:

```
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

Or install and run Redis locally according to the instructions for your operating system.
"""
logger.info("Make sure you have a Redis server running. You can start one using Docker with the following command:")


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
logger.debug(f"Connecting to Redis at: {REDIS_URL}")

"""
## Importing Required Libraries
"""
logger.info("## Importing Required Libraries")


"""
## Basic Usage of RedisChatMessageHistory
"""
logger.info("## Basic Usage of RedisChatMessageHistory")

history = RedisChatMessageHistory(session_id="user_123", redis_url=REDIS_URL)

history.add_user_message("Hello, AI assistant!")
history.add_ai_message("Hello! How can I assist you today?")

logger.debug("Chat History:")
for message in history.messages:
    logger.debug(f"{type(message).__name__}: {message.content}")

"""
## Using RedisChatMessageHistory with Language Models

### Set Ollama API key
"""
logger.info("## Using RedisChatMessageHistory with Language Models")

# from getpass import getpass

# ollama_api_key = os.getenv("OPENAI_API_KEY")

if not ollama_api_key:
    logger.debug("Ollama API key not found in environment variables.")
#     ollama_api_key = getpass("Please enter your Ollama API key: ")

#     os.environ["OPENAI_API_KEY"] = ollama_api_key
    logger.debug("Ollama API key has been set for this session.")
else:
    logger.debug("Ollama API key found in environment variables.")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

llm = ChatOllama(model="llama3.2")

chain = prompt | llm


def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(session_id, redis_url=REDIS_URL)


chain_with_history = RunnableWithMessageHistory(
    chain, get_redis_history, input_messages_key="input", history_messages_key="history"
)

response1 = chain_with_history.invoke(
    {"input": "Hi, my name is Alice."},
    config={"configurable": {"session_id": "alice_123"}},
)
logger.debug("AI Response 1:", response1.content)

response2 = chain_with_history.invoke(
    {"input": "What's my name?"}, config={"configurable": {"session_id": "alice_123"}}
)
logger.debug("AI Response 2:", response2.content)

"""
## Advanced Features

### Custom Redis Configuration
"""
logger.info("## Advanced Features")

custom_history = RedisChatMessageHistory(
    "user_456",
    redis_url=REDIS_URL,
    key_prefix="custom_prefix:",
    ttl=3600,  # Set TTL to 1 hour
    index_name="custom_index",
)

custom_history.add_user_message("This is a message with custom configuration.")
logger.debug("Custom History:", custom_history.messages)

"""
### Searching Messages
"""
logger.info("### Searching Messages")

history.add_user_message("Tell me about artificial intelligence.")
history.add_ai_message(
    "Artificial Intelligence (AI) is a branch of computer science..."
)

search_results = history.search_messages("artificial intelligence")
logger.debug("Search Results:")
for result in search_results:
    logger.debug(f"{result['type']}: {result['content'][:50]}...")

"""
### Clearing History
"""
logger.info("### Clearing History")

history.clear()
logger.debug("Messages after clearing:", history.messages)

"""
## Conclusion

This notebook demonstrated the key features of `RedisChatMessageHistory` from the langchain-redis package. It showed how to initialize and use the chat history, integrate it with language models, and utilize advanced features like custom configurations and message searching. Redis provides a fast and scalable solution for managing chat history in AI applications.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)
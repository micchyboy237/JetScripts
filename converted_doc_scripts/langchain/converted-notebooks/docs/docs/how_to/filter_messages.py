from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import (
AIMessage,
HumanMessage,
SystemMessage,
filter_messages,
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
# How to filter messages

In more complex chains and agents we might track state with a list of [messages](/docs/concepts/messages/). This list can start to accumulate messages from multiple different models, speakers, sub-chains, etc., and we may only want to pass subsets of this full list of messages to each model call in the chain/agent.

The `filter_messages` utility makes it easy to filter messages by type, id, or name.

## Basic usage
"""
logger.info("# How to filter messages")


messages = [
    SystemMessage("you are a good assistant", id="1"),
    HumanMessage("example input", id="2", name="example_user"),
    AIMessage("example output", id="3", name="example_assistant"),
    HumanMessage("real input", id="4", name="bob"),
    AIMessage("real output", id="5", name="alice"),
]

filter_messages(messages, include_types="human")

filter_messages(messages, exclude_names=["example_user", "example_assistant"])

filter_messages(messages, include_types=[HumanMessage, AIMessage], exclude_ids=["3"])

"""
## Chaining

`filter_messages` can be used imperatively (like above) or declaratively, making it easy to compose with other components in a chain:
"""
logger.info("## Chaining")

# %pip install -qU langchain-anthropic


llm = ChatOllama(model="llama3.2")
filter_ = filter_messages(exclude_names=["example_user", "example_assistant"])
chain = filter_ | llm
chain.invoke(messages)

"""
Looking at the LangSmith trace we can see that before the messages are passed to the model they are filtered: https://smith.langchain.com/public/f808a724-e072-438e-9991-657cc9e7e253/r

Looking at just the filter_, we can see that it's a Runnable object that can be invoked like all Runnables:
"""
logger.info("Looking at the LangSmith trace we can see that before the messages are passed to the model they are filtered: https://smith.langchain.com/public/f808a724-e072-438e-9991-657cc9e7e253/r")

filter_.invoke(messages)

"""
## API reference

For a complete description of all arguments head to the API reference: https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.filter_messages.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
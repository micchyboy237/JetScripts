from jet.llm.ollama.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.agent import ReActAgent
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.tools import FunctionTool
from llama_index.memory.mem0 import Mem0Memory
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: LlamaIndex
---

LlamaIndex supports Mem0 as a [memory store](https://llamahub.ai/l/memory/llama-index-memory-mem0). In this guide, we'll show you how to use it.

<Note type="info">
  🎉 Exciting news! [**Mem0Memory**](https://docs.llamaindex.ai/en/stable/examples/memory/Mem0Memory/) now supports **ReAct** and **FunctionCalling** agents.
</Note>

### Installation

To install the required package, run:
"""
logger.info("### Installation")

pip install llama-index-core llama-index-memory-mem0

"""
### Setup with Mem0 Platform

Set your Mem0 Platform API key as an environment variable. You can replace `<your-mem0-api-key>` with your actual API key:

<Note type="info">
  You can obtain your Mem0 Platform API key from the [Mem0 Platform](https://app.mem0.ai/login).
</Note>
"""
logger.info("### Setup with Mem0 Platform")

os.environ["MEM0_API_KEY"] = "<your-mem0-api-key>"

"""
Import the necessary modules and create a Mem0Memory instance:
"""
logger.info("Import the necessary modules and create a Mem0Memory instance:")


context = {"user_id": "user_1"}
memory_from_client = Mem0Memory.from_client(
    context=context,
    api_key="<your-mem0-api-key>",
    search_msg_limit=4,  # optional, default is 5
)

"""
Context is used to identify the user, agent or the conversation in the Mem0. It is required to be passed in the at least one of the fields in the `Mem0Memory` constructor. It can be any of the following:
"""
logger.info("Context is used to identify the user, agent or the conversation in the Mem0. It is required to be passed in the at least one of the fields in the `Mem0Memory` constructor. It can be any of the following:")

context = {
    "user_id": "user_1",
    "agent_id": "agent_1",
    "run_id": "run_1",
}

"""
`search_msg_limit` is optional, default is 5. It is the number of messages from the chat history to be used for memory retrieval from Mem0. More number of messages will result in more context being used for retrieval but will also increase the retrieval time and might result in some unwanted results.

<Note type="info">
  `search_msg_limit` is different from `limit`. `limit` is the number of messages to be retrieved from Mem0 and is used in search.
</Note>

### Setup with Mem0 OSS

Set your Mem0 OSS by providing configuration details:

<Note type="info">
  To know more about Mem0 OSS, read [Mem0 OSS Quickstart](https://docs.mem0.ai/open-source/overview).
</Note>
"""
logger.info("### Setup with Mem0 OSS")

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test_9",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 1536,  # Change this according to your local model's dimensions
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 2000,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "mxbai-embed-large"},
    },
    "version": "v1.1",
}

"""
Create a Mem0Memory instance:
"""
logger.info("Create a Mem0Memory instance:")

memory_from_config = Mem0Memory.from_config(
    context=context,
    config=config,
    search_msg_limit=4,  # optional, default is 5
)

"""
Initialize the LLM
"""
logger.info("Initialize the LLM")


# os.environ["OPENAI_API_KEY"] = "<your-openai-api-key>"
llm = MLX(model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats")

"""
### SimpleChatEngine
Use the `SimpleChatEngine` to start a chat with the agent with the memory.
"""
logger.info("### SimpleChatEngine")


agent = SimpleChatEngine.from_defaults(
    llm=llm, memory=memory_from_client  # or memory_from_config
)

response = agent.chat("Hi, My name is Mayank")
logger.debug(response)

"""
Now we will learn how to use Mem0 with FunctionCalling and ReAct agents.

Initialize the tools:
"""
logger.info("Now we will learn how to use Mem0 with FunctionCalling and ReAct agents.")



def call_fn(name: str):
    """Call the provided name.
    Args:
        name: str (Name of the person)
    """
    logger.debug(f"Calling... {name}")


def email_fn(name: str):
    """Email the provided name.
    Args:
        name: str (Name of the person)
    """
    logger.debug(f"Emailing... {name}")


call_tool = FunctionTool.from_defaults(fn=call_fn)
email_tool = FunctionTool.from_defaults(fn=email_fn)

"""
### FunctionCallingAgent
"""
logger.info("### FunctionCallingAgent")


agent = FunctionCallingAgent.from_tools(
    [call_tool, email_tool],
    llm=llm,
    memory=memory_from_client,  # or memory_from_config
    verbose=True,
)

response = agent.chat("Hi, My name is Mayank")
logger.debug(response)

"""
### ReActAgent
"""
logger.info("### ReActAgent")


agent = ReActAgent.from_tools(
    [call_tool, email_tool],
    llm=llm,
    memory=memory_from_client,  # or memory_from_config
    verbose=True,
)

response = agent.chat("Hi, My name is Mayank")
logger.debug(response)

"""
## Key Features

1. **Memory Integration**: Uses Mem0 to store and retrieve relevant information from past interactions.
2. **Personalization**: Provides context-aware agent responses based on user history and preferences.
3. **Flexible Architecture**: LlamaIndex allows for easy integration of the memory with the agent.
4. **Continuous Learning**: Each interaction is stored, improving future responses.

## Conclusion

By integrating LlamaIndex with Mem0, you can build a personalized agent that can maintain context across interactions with the agent and provide tailored recommendations and assistance.

## Help

- For more details on LlamaIndex, visit the [LlamaIndex documentation](https://llamahub.ai/l/memory/llama-index-memory-mem0).
- [Mem0 Platform](https://app.mem0.ai/).
- If you need further assistance, please feel free to reach out to us through following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Key Features")

logger.info("\n\n[DONE]", bright=True)
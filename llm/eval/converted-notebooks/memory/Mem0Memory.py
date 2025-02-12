from llama_index.core.agent import ReActAgent
from llama_index.core.chat_engine.simple import SimpleChatEngine
import nest_asyncio
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from jet.llm.ollama.base import Ollama
from llama_index.memory.mem0 import Mem0Memory
import os
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/memory/Mem0Memory.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Mem0
#
# Mem0 (pronounced “mem-zero”) enhances AI assistants and agents with an intelligent memory layer, enabling personalized AI interactions. It remembers user preferences and traits and continuously updates over time, making it ideal for applications like customer support chatbots and AI assistants.
#
# Mem0 offers two powerful ways to leverage our technology: our [managed platform](https://docs.mem0.ai/platform/overview) and our [open source solution](https://docs.mem0.ai/open-source/quickstart).

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %pip install llama-index-memory-mem0

# !pip install llama-index

# Setup with Mem0 Platform
#
# Set your Mem0 Platform API key as an environment variable. You can replace `<your-mem0-api-key>` with your actual API key:
#
# > Note: You can obtain your Mem0 Platform API key from the [Mem0 Platform](https://app.mem0.ai/login).


# os.environ["MEM0_API_KEY"] = "<your-mem0-api-key>"

# Using `from_client` (for Mem0 platform API):


context = {"user_id": "test_user_1"}
memory_from_client = Mem0Memory.from_client(
    context=context,
    # api_key="<your-api-key>",
    search_msg_limit=4,  # Default is 5
)

# Mem0 Context is used to identify the user, agent or the conversation in the Mem0. It is required to be passed in the at least one of the fields in the `Mem0Memory` constructor.
#
# `search_msg_limit` is optional, default is 5. It is the number of messages from the chat history to be used for memory retrieval from Mem0. More number of messages will result in more context being used for retrieval but will also increase the retrieval time and might result in some unwanted results.

# Using `from_config` (for Mem0 OSS)

# os.environ["OPENAI_API_KEY"] = "<your-api-key>"
config = {
    # "vector_store": {
    #     "provider": "qdrant",
    #     "config": {
    #         "collection_name": "test_9",
    #         "host": "localhost",
    #         "port": 6333,
    #         "embedding_model_dims": 1024,  # Change this according to your local model's dimensions
    #     },
    # },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1",
            "temperature": 0.2,
            "max_tokens": 1500,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {"model": "mxbai-embed-large"},
    },
    "version": "v1.1",
}
memory_from_config = Mem0Memory.from_config(
    context=context,
    config=config,
    search_msg_limit=4,  # Default is 5
)

# Initialize LLM


llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)

# Mem0 for Function Calling Agents
#
# Use `Mem0` as memory for `FunctionCallingAgents`.


nest_asyncio.apply()

# Initialize Tools


def call_fn(name: str):
    """Call the provided name.
    Args:
        name: str (Name of the person)
    """
    print(f"Calling... {name}")


def email_fn(name: str):
    """Email the provided name.
    Args:
        name: str (Name of the person)
    """
    print(f"Emailing... {name}")


call_tool = FunctionTool.from_defaults(fn=call_fn)
email_tool = FunctionTool.from_defaults(fn=email_fn)

agent = FunctionCallingAgent.from_tools(
    [call_tool, email_tool],
    llm=llm,
    memory=memory_from_client,  # can be memory_from_config
    verbose=True,
)

response = agent.chat("Hi, My name is Mayank.")

response = agent.chat("My preferred way of communication would be Email.")

response = agent.chat("Send me an update of your product.")

# Mem0 for Chat Engines
#
# Use `Mem0` as memory to `SimpleChatEngine`.


agent = SimpleChatEngine.from_defaults(
    llm=llm, memory=memory_from_client  # can be memory_from_config
)

response = agent.chat("Hi, My name is mayank")
print(response)

response = agent.chat("I am planning to visit SF tommorow.")
print(response)

response = agent.chat(
    "What would be a suitable time to schedule a meeting tommorow?"
)
print(response)

# Mem0 for ReAct Agents
#
# Use `Mem0` as memory for `ReActAgent`.


agent = ReActAgent.from_tools(
    [call_tool, email_tool],
    llm=llm,
    memory=memory_from_client,  # can be memory_from_config
    verbose=True,
)

response = agent.chat("Hi, My name is Mayank.")

response = agent.chat("My preferred way of communication would be Email.")

response = agent.chat("Send me an update of your product.")

response = agent.chat("First call me and then communicate me requirements.")

logger.info("\n\n[DONE]", bright=True)

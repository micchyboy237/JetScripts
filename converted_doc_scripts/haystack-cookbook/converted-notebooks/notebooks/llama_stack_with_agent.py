from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack_integrations.components.generators.llama_stack import LlamaStackChatGenerator
from jet.logger import logger
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
# 🛠️🦙 Build with Llama Stack and Haystack Agent


This notebook demonstrates how to use the `LlamaStackChatGenerator` component with Haystack [Agent](https://docs.haystack.deepset.ai/docs/agent) to enable function calling capabilities. We'll create a simple weather tool that the `Agent` can call to provide dynamic, up-to-date information.

We start with installing integration package.
"""
logger.info("# 🛠️🦙 Build with Llama Stack and Haystack Agent")

# %%bash

pip install llama-stack-haystack

"""
## Setup

Before running this example, you need to:

1. Set up Llama Stack Server through an inference provider
2. Have a model available (e.g., `llama3.2:3b`)

For a quick start on how to setup server with Ollama, see the [Llama Stack documentation](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html).

Once you have the server running, it will typically be available at `http://localhost:8321/v1/ollama/v1`.

## Defining a Tool

[Tool](https://docs.haystack.deepset.ai/docs/tool) in Haystack allow models to call functions to get real-time information or perform actions. Let's create a simple weather tool that the model can use to provide weather information.
"""
logger.info("## Setup")


def weather(city: str):
    """Return mock weather info for the given city."""
    return f"The weather in {city} is sunny and 32°C"

tool_parameters = {
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    },
    "required": ["city"]
}

weather_tool = Tool(
    name="weather",
    description="Useful for getting the weather in a specific city",
    parameters=tool_parameters,
    function=weather,
)

"""
## Setting Up Agent

Now, let's create a `LlamaStackChatGenerator` and pass it to the `Agent`. The Agent component will use the model running with `LlamaStackChatGenerator` to reason and make decisions.
"""
logger.info("## Setting Up Agent")


chat_generator = LlamaStackChatGenerator(
    model="ollama/llama3.2:3b",  # model name varies depending on the inference provider used for the Llama Stack Server
    api_base_url="http://localhost:8321/v1/ollama/v1",
)
agent = Agent(
    chat_generator=chat_generator,
    tools=[weather_tool],
)

agent.warm_up()

"""
## Using Tools with the Agent

Now, when we ask questions, the `Agent` will utilize both the provided `tool` and the `LlamaStackChatGenerator` to generate answers. We enable the streaming in `Agent` through `streaming_callback`, so you can observe the tool calls and results in real time.
"""
logger.info("## Using Tools with the Agent")

messages = [ChatMessage.from_user("What's the weather in Tokyo?")]

response = agent.run(messages=messages, tools=[weather_tool],     streaming_callback=print_streaming_chunk,
)

"""
## Simple Chat with ChatGenerator
For a simpler use case, you can also create a lightweight mechanism to chat directly with `LlamaStackChatGenerator`.
"""
logger.info("## Simple Chat with ChatGenerator")

messages = []

while True:
  msg = input("Enter your message or Q to exit\n🧑 ")
  if msg=="Q":
    break
  messages.append(ChatMessage.from_user(msg))
  response = chat_generator.run(messages=messages)
  assistant_resp = response['replies'][0]
  logger.debug("🤖 "+assistant_resp.text)
  messages.append(assistant_resp)

"""
If you want to switch your model provider, you can reuse the same `LlamaStackChatGenerator` code with different providers. Simply run the desired inference provider on the Llama Stack Server and update the `model` name during the initialization of `LlamaStackChatGenerator`.

For more details on available inference providers, see [Llama Stack docs](https://llama-stack.readthedocs.io/en/latest/providers/inference/index.html).
"""
logger.info("If you want to switch your model provider, you can reuse the same `LlamaStackChatGenerator` code with different providers. Simply run the desired inference provider on the Llama Stack Server and update the `model` name during the initialization of `LlamaStackChatGenerator`.")

logger.info("\n\n[DONE]", bright=True)
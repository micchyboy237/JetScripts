from jet.adapters.langchain.chat_ollama.chat_models import ChatOllama
from jet.logger import logger
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from operator import itemgetter
import ChatModelTabs from "@theme/ChatModelTabs";
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
---
sidebar_position: 0
---

# How to use tools in a chain

In this guide, we will go over the basic ways to create Chains and Agents that call [Tools](/docs/concepts/tools/). Tools can be just about anything — APIs, functions, databases, etc. Tools allow us to extend the capabilities of a model beyond just outputting text/messages. The key to using models with tools is correctly prompting a model and parsing its response so that it chooses the right tools and provides the right inputs for them.

## Setup

We'll need to install the following packages for this guide:
"""
logger.info("# How to use tools in a chain")

# %pip install --upgrade --quiet langchain

"""
If you'd like to trace your runs in [LangSmith](https://docs.smith.langchain.com/) uncomment and set the following environment variables:
"""
logger.info("If you'd like to trace your runs in [LangSmith](https://docs.smith.langchain.com/) uncomment and set the following environment variables:")

# import getpass

"""
## Create a tool

First, we need to create a tool to call. For this example, we will create a custom tool from a function. For more information on creating custom tools, please see [this guide](/docs/how_to/custom_tools).
"""
logger.info("## Create a tool")



@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

logger.debug(multiply.name)
logger.debug(multiply.description)
logger.debug(multiply.args)

multiply.invoke({"first_int": 4, "second_int": 5})

"""
## Chains

If we know that we only need to use a tool a fixed number of times, we can create a chain for doing so. Let's create a simple chain that just multiplies user-specified numbers.

![chain](../../static/img/tool_chain.svg)

### Tool/function calling
One of the most reliable ways to use tools with LLMs is with [tool calling](/docs/concepts/tool_calling/) APIs (also sometimes called function calling). This only works with models that explicitly support tool calling. You can see which models support tool calling [here](/docs/integrations/chat/), and learn more about how to use tool calling in [this guide](/docs/how_to/function_calling).

First we'll define our model and tools. We'll start with just a single tool, `multiply`.


<ChatModelTabs customVarName="llm"/>
"""
logger.info("## Chains")


llm = ChatOllama(model="llama3.2")

"""
We'll use `bind_tools` to pass the definition of our tool in as part of each call to the model, so that the model can invoke the tool when appropriate:
"""
logger.info("We'll use `bind_tools` to pass the definition of our tool in as part of each call to the model, so that the model can invoke the tool when appropriate:")

llm_with_tools = llm.bind_tools([multiply])

"""
When the model invokes the tool, this will show up in the `AIMessage.tool_calls` attribute of the output:
"""
logger.info("When the model invokes the tool, this will show up in the `AIMessage.tool_calls` attribute of the output:")

msg = llm_with_tools.invoke("whats 5 times forty two")
msg.tool_calls

"""
Check out the [LangSmith trace here](https://smith.langchain.com/public/81ff0cbd-e05b-4720-bf61-2c9807edb708/r).

### Invoking the tool

Great! We're able to generate tool invocations. But what if we want to actually call the tool? To do so we'll need to pass the generated tool args to our tool. As a simple example we'll just extract the arguments of the first tool_call:
"""
logger.info("### Invoking the tool")


chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]) | multiply
chain.invoke("What's four times 23")

"""
Check out the [LangSmith trace here](https://smith.langchain.com/public/16bbabb9-fc9b-41e5-a33d-487c42df4f85/r).

## Agents

Chains are great when we know the specific sequence of tool usage needed for any user input. But for certain use cases, how many times we use tools depends on the input. In these cases, we want to let the model itself decide how many times to use tools and in what order. [Agents](/docs/concepts/agents/) let us do just this.

We'll demonstrate a simple example using a LangGraph agent. See [this tutorial](/docs/tutorials/agents) for more detail.

![agent](../../static/img/tool_agent.svg)
"""
logger.info("## Agents")

# !pip install -qU langgraph


"""
Agents are also great because they make it easy to use multiple tools.
"""
logger.info("Agents are also great because they make it easy to use multiple tools.")

@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent


tools = [multiply, add, exponentiate]

agent = create_react_agent(llm, tools)

"""
With an agent, we can ask questions that require arbitrarily-many uses of our tools:
"""
logger.info("With an agent, we can ask questions that require arbitrarily-many uses of our tools:")

query = (
    "Take 3 to the fifth power and multiply that by the sum of twelve and "
    "three, then square the whole result."
)
input_message = {"role": "user", "content": query}

for step in agent.stream({"messages": [input_message]}, stream_mode="values"):
    step["messages"][-1].pretty_logger.debug()

"""
Check out the [LangSmith trace here](https://smith.langchain.com/public/eeeb27a4-a2f8-4f06-a3af-9c983f76146c/r).
"""
logger.info("Check out the [LangSmith trace here](https://smith.langchain.com/public/eeeb27a4-a2f8-4f06-a3af-9c983f76146c/r).")

logger.info("\n\n[DONE]", bright=True)
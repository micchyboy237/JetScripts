from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
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
# How to pass tool outputs to chat models

:::info Prerequisites
This guide assumes familiarity with the following concepts:

- [LangChain Tools](/docs/concepts/tools)
- [Function/tool calling](/docs/concepts/tool_calling)
- [Using chat models to call tools](/docs/how_to/tool_calling)
- [Defining custom tools](/docs/how_to/custom_tools/)

:::

Some models are capable of [**tool calling**](/docs/concepts/tool_calling) - generating arguments that conform to a specific user-provided schema. This guide will demonstrate how to use those tool calls to actually call a function and properly pass the results back to the model.

![Diagram of a tool call invocation](/img/tool_invocation.png)

![Diagram of a tool call result](/img/tool_results.png)

First, let's define our tools and our model:


<ChatModelTabs
  customVarName="llm"
  overrideParams={{fireworks: {model: "accounts/fireworks/models/firefunction-v1", kwargs: "temperature=0"}}}
/>
"""
logger.info("# How to pass tool outputs to chat models")

# from getpass import getpass


# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

llm = ChatOllama(model="llama3.2")



@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

llm_with_tools = llm.bind_tools(tools)

"""
Now, let's get the model to call a tool. We'll add it to a list of messages that we'll treat as conversation history:
"""
logger.info("Now, let's get the model to call a tool. We'll add it to a list of messages that we'll treat as conversation history:")


query = "What is 3 * 12? Also, what is 11 + 49?"

messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)

logger.debug(ai_msg.tool_calls)

messages.append(ai_msg)

"""
Next let's invoke the tool functions using the args the model populated!

Conveniently, if we invoke a LangChain `Tool` with a `ToolCall`, we'll automatically get back a `ToolMessage` that can be fed back to the model:

:::caution Compatibility

This functionality was added in `langchain-core == 0.2.19`. Please make sure your package is up to date.

If you are on earlier versions of `langchain-core`, you will need to extract the `args` field from the tool and construct a `ToolMessage` manually.

:::
"""
logger.info("Next let's invoke the tool functions using the args the model populated!")

for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

messages

"""
And finally, we'll invoke the model with the tool results. The model will use this information to generate a final answer to our original query:
"""
logger.info("And finally, we'll invoke the model with the tool results. The model will use this information to generate a final answer to our original query:")

llm_with_tools.invoke(messages)

"""
Note that each `ToolMessage` must include a `tool_call_id` that matches an `id` in the original tool calls that the model generates. This helps the model match tool responses with tool calls.

Tool calling agents, like those in [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/), use this basic flow to answer queries and solve tasks.

## Related

- [LangGraph quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- Few shot prompting [with tools](/docs/how_to/tools_few_shot/)
- Stream [tool calls](/docs/how_to/tool_streaming/)
- Pass [runtime values to tools](/docs/how_to/tool_runtime)
- Getting [structured outputs](/docs/how_to/structured_output/) from models
"""
logger.info("## Related")

logger.info("\n\n[DONE]", bright=True)
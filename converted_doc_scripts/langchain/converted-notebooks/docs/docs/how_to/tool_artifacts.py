from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.tools import BaseTool
from langchain_core.tools import tool
from operator import attrgetter
from typing import List, Tuple
import ChatModelTabs from "@theme/ChatModelTabs";
import os
import random
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
# How to return artifacts from a tool

:::info Prerequisites
This guide assumes familiarity with the following concepts:

- [ToolMessage](/docs/concepts/messages/#toolmessage)
- [Tools](/docs/concepts/tools)
- [Function/tool calling](/docs/concepts/tool_calling)

:::

[Tools](/docs/concepts/tools/) are utilities that can be [called by a model](/docs/concepts/tool_calling/), and whose outputs are designed to be fed back to a model. Sometimes, however, there are artifacts of a tool's execution that we want to make accessible to downstream components in our chain or agent, but that we don't want to expose to the model itself. For example if a tool returns a custom object, a dataframe or an image, we may want to pass some metadata about this output to the model without passing the actual output to the model. At the same time, we may want to be able to access this full output elsewhere, for example in downstream tools.

The Tool and [ToolMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.tool.ToolMessage.html) interfaces make it possible to distinguish between the parts of the tool output meant for the model (this is the ToolMessage.content) and those parts which are meant for use outside the model (ToolMessage.artifact).

:::info Requires ``langchain-core >= 0.2.19``

This functionality was added in ``langchain-core == 0.2.19``. Please make sure your package is up to date.

:::

## Defining the tool

If we want our tool to distinguish between message content and other artifacts, we need to specify `response_format="content_and_artifact"` when defining our tool and make sure that we return a tuple of (content, artifact):
"""
logger.info("# How to return artifacts from a tool")

# %pip install -qU "langchain-core>=0.2.19"




@tool(response_format="content_and_artifact")
def generate_random_ints(min: int, max: int, size: int) -> Tuple[str, List[int]]:
    """Generate size random ints in the range [min, max]."""
    array = [random.randint(min, max) for _ in range(size)]
    content = f"Successfully generated array of {size} random ints in [{min}, {max}]."
    return content, array

"""
## Invoking the tool with ToolCall

If we directly invoke our tool with just the tool arguments, you'll notice that we only get back the content part of the Tool output:
"""
logger.info("## Invoking the tool with ToolCall")

generate_random_ints.invoke({"min": 0, "max": 9, "size": 10})

"""
In order to get back both the content and the artifact, we need to invoke our model with a ToolCall (which is just a dictionary with "name", "args", "id" and "type" keys), which has additional info needed to generate a ToolMessage like the tool call ID:
"""
logger.info("In order to get back both the content and the artifact, we need to invoke our model with a ToolCall (which is just a dictionary with "name", "args", "id" and "type" keys), which has additional info needed to generate a ToolMessage like the tool call ID:")

generate_random_ints.invoke(
    {
        "name": "generate_random_ints",
        "args": {"min": 0, "max": 9, "size": 10},
        "id": "123",  # required
        "type": "tool_call",  # required
    }
)

"""
## Using with a model

With a [tool-calling model](/docs/how_to/tool_calling/), we can easily use a model to call our Tool and generate ToolMessages:


<ChatModelTabs
  customVarName="llm"
/>
"""
logger.info("## Using with a model")


llm = ChatOllama(model="llama3.2")

llm_with_tools = llm.bind_tools([generate_random_ints])

ai_msg = llm_with_tools.invoke("generate 6 positive ints less than 25")
ai_msg.tool_calls

generate_random_ints.invoke(ai_msg.tool_calls[0])

"""
If we just pass in the tool call args, we'll only get back the content:
"""
logger.info("If we just pass in the tool call args, we'll only get back the content:")

generate_random_ints.invoke(ai_msg.tool_calls[0]["args"])

"""
If we wanted to declaratively create a chain, we could do this:
"""
logger.info("If we wanted to declaratively create a chain, we could do this:")


chain = llm_with_tools | attrgetter("tool_calls") | generate_random_ints.map()

chain.invoke("give me a random number between 1 and 5")

"""
## Creating from BaseTool class

If you want to create a BaseTool object directly, instead of decorating a function with `@tool`, you can do so like this:
"""
logger.info("## Creating from BaseTool class")



class GenerateRandomFloats(BaseTool):
    name: str = "generate_random_floats"
    description: str = "Generate size random floats in the range [min, max]."
    response_format: str = "content_and_artifact"

    ndigits: int = 2

    def _run(self, min: float, max: float, size: int) -> Tuple[str, List[float]]:
        range_ = max - min
        array = [
            round(min + (range_ * random.random()), ndigits=self.ndigits)
            for _ in range(size)
        ]
        content = f"Generated {size} floats in [{min}, {max}], rounded to {self.ndigits} decimals."
        return content, array

rand_gen = GenerateRandomFloats(ndigits=4)
rand_gen.invoke({"min": 0.1, "max": 3.3333, "size": 3})

rand_gen.invoke(
    {
        "name": "generate_random_floats",
        "args": {"min": 0.1, "max": 3.3333, "size": 3},
        "id": "123",
        "type": "tool_call",
    }
)

logger.info("\n\n[DONE]", bright=True)
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.ollama_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from pydantic import BaseModel, Field
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
sidebar_position: 2
---

# How to do tool/function calling

:::info
We use the term tool calling interchangeably with function calling. Although
function calling is sometimes meant to refer to invocations of a single function,
we treat all models as though they can return multiple tool or function calls in 
each message.
:::

Tool calling allows a model to respond to a given prompt by generating output that 
matches a user-defined schema. While the name implies that the model is performing 
some action, this is actually not the case! The model is coming up with the 
arguments to a tool, and actually running the tool (or not) is up to the user - 
for example, if you want to [extract output matching some schema](/docs/tutorials/extraction) 
from unstructured text, you could give the model an "extraction" tool that takes 
parameters matching the desired schema, then treat the generated output as your final 
result.

A tool call includes a name, arguments dict, and an optional identifier. The 
arguments dict is structured `{argument_name: argument_value}`.

Many LLM providers, including [Ollama](https://www.anthropic.com/), 
[Cohere](https://cohere.com/), [Google](https://cloud.google.com/vertex-ai), 
[Mistral](https://mistral.ai/), [Ollama](https://ollama.com/), and others, 
support variants of a tool calling feature. These features typically allow requests 
to the LLM to include available tools and their schemas, and for responses to include 
calls to these tools. For instance, given a search engine tool, an LLM might handle a 
query by first issuing a call to the search engine. The system calling the LLM can 
receive the tool call, execute it, and return the output to the LLM to inform its 
response. LangChain includes a suite of [built-in tools](/docs/integrations/tools/) 
and supports several methods for defining your own [custom tools](/docs/how_to/custom_tools). 
Tool-calling is extremely useful for building [tool-using chains and agents](/docs/how_to#tools), 
and for getting structured outputs from models more generally.

Providers adopt different conventions for formatting tool schemas and tool calls. 
For instance, Ollama returns tool calls as parsed structures within a larger content block:
```python
[
  {
    "text": "<thinking>\nI should use a tool.\n</thinking>",
    "type": "text"
  },
  {
    "id": "id_value",
    "input": {"arg_name": "arg_value"},
    "name": "tool_name",
    "type": "tool_use"
  }
]
```
whereas Ollama separates tool calls into a distinct parameter, with arguments as JSON strings:
```python
{
  "tool_calls": [
    {
      "id": "id_value",
      "function": {
        "arguments": '{"arg_name": "arg_value"}',
        "name": "tool_name"
      },
      "type": "function"
    }
  ]
}
```
LangChain implements standard interfaces for defining tools, passing them to LLMs, 
and representing tool calls.

## Passing tools to LLMs

Chat models supporting tool calling features implement a `.bind_tools` method, which 
receives a list of LangChain [tool objects](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.BaseTool.html#langchain_core.tools.BaseTool) 
and binds them to the chat model in its expected format. Subsequent invocations of the 
chat model will include tool schemas in its calls to the LLM.

For example, we can define the schema for custom tools using the `@tool` decorator 
on Python functions:
"""
logger.info("# How to do tool/function calling")



@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

"""
Or below, we define the schema using Pydantic:
"""
logger.info("Or below, we define the schema using Pydantic:")



class Add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


tools = [Add, Multiply]

"""
We can bind them to chat models as follows:


<ChatModelTabs
  customVarName="llm"
  overrideParams={{fireworks: {model: "accounts/fireworks/models/firefunction-v1", kwargs: "temperature=0"}}}
/>

We can use the `bind_tools()` method to handle converting
`Multiply` to a "tool" and binding it to the model (i.e.,
passing it in each time the model is invoked).
"""
logger.info("We can bind them to chat models as follows:")


llm = ChatOllama(model="llama3.2")

llm_with_tools = llm.bind_tools(tools)

"""
## Tool calls

If tool calls are included in a LLM response, they are attached to the corresponding 
[message](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage) 
or [message chunk](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk) 
as a list of [tool call](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.tool.ToolCall.html#langchain_core.messages.tool.ToolCall) 
objects in the `.tool_calls` attribute. A `ToolCall` is a typed dict that includes a 
tool name, dict of argument values, and (optionally) an identifier. Messages with no 
tool calls default to an empty list for this attribute.

Example:
"""
logger.info("## Tool calls")

query = "What is 3 * 12? Also, what is 11 + 49?"

llm_with_tools.invoke(query).tool_calls

"""
The `.tool_calls` attribute should contain valid tool calls. Note that on occasion, 
model providers may output malformed tool calls (e.g., arguments that are not 
valid JSON). When parsing fails in these cases, instances 
of [InvalidToolCall](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.tool.InvalidToolCall.html#langchain_core.messages.tool.InvalidToolCall) 
are populated in the `.invalid_tool_calls` attribute. An `InvalidToolCall` can have 
a name, string arguments, identifier, and error message.

If desired, [output parsers](/docs/how_to#output-parsers) can further 
process the output. For example, we can convert back to the original Pydantic class:
"""
logger.info("The `.tool_calls` attribute should contain valid tool calls. Note that on occasion,")


chain = llm_with_tools | PydanticToolsParser(tools=[Multiply, Add])
chain.invoke(query)

"""
### Streaming

When tools are called in a streaming context, 
[message chunks](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk) 
will be populated with [tool call chunk](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.tool.ToolCallChunk.html#langchain_core.messages.tool.ToolCallChunk) 
objects in a list via the `.tool_call_chunks` attribute. A `ToolCallChunk` includes 
optional string fields for the tool `name`, `args`, and `id`, and includes an optional 
integer field `index` that can be used to join chunks together. Fields are optional 
because portions of a tool call may be streamed across different chunks (e.g., a chunk 
that includes a substring of the arguments may have null values for the tool name and id).

Because message chunks inherit from their parent message class, an 
[AIMessageChunk](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk) 
with tool call chunks will also include `.tool_calls` and `.invalid_tool_calls` fields. 
These fields are parsed best-effort from the message's tool call chunks.

Note that not all providers currently support streaming for tool calls.

Example:
"""
logger.info("### Streaming")

for chunk in llm_with_tools.stream(query):
    logger.debug(chunk.tool_call_chunks)

"""
Note that adding message chunks will merge their corresponding tool call chunks. This is the principle by which LangChain's various [tool output parsers](/docs/how_to/output_parser_structured) support streaming.

For example, below we accumulate tool call chunks:
"""
logger.info("Note that adding message chunks will merge their corresponding tool call chunks. This is the principle by which LangChain's various [tool output parsers](/docs/how_to/output_parser_structured) support streaming.")

first = True
for chunk in llm_with_tools.stream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk

    logger.debug(gathered.tool_call_chunks)

logger.debug(type(gathered.tool_call_chunks[0]["args"]))

"""
And below we accumulate tool calls to demonstrate partial parsing:
"""
logger.info("And below we accumulate tool calls to demonstrate partial parsing:")

first = True
for chunk in llm_with_tools.stream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk

    logger.debug(gathered.tool_calls)

logger.debug(type(gathered.tool_calls[0]["args"]))

"""
## Passing tool outputs to model

If we're using the model-generated tool invocations to actually call tools and want to pass the tool results back to the model, we can do so using `ToolMessage`s.
"""
logger.info("## Passing tool outputs to model")


messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
messages

llm_with_tools.invoke(messages)

"""
## Few-shot prompting

For more complex tool use it's very useful to add few-shot examples to the prompt. We can do this by adding `AIMessage`s with `ToolCall`s and corresponding `ToolMessage`s to our prompt.

For example, even with some special instructions our model can get tripped up by order of operations:
"""
logger.info("## Few-shot prompting")

llm_with_tools.invoke(
    "Whats 119 times 8 minus 20. Don't do any math yourself, only use tools for math. Respect order of operations"
).tool_calls

"""
The model shouldn't be trying to add anything yet, since it technically can't know the results of 119 * 8 yet.

By adding a prompt with some examples we can correct this behavior:
"""
logger.info("The model shouldn't be trying to add anything yet, since it technically can't know the results of 119 * 8 yet.")


examples = [
    HumanMessage(
        "What's the product of 317253 and 128472 plus four", name="example_user"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {"name": "Multiply", "args": {"x": 317253, "y": 128472}, "id": "1"}
        ],
    ),
    ToolMessage("16505054784", tool_call_id="1"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{"name": "Add", "args": {"x": 16505054784, "y": 4}, "id": "2"}],
    ),
    ToolMessage("16505054788", tool_call_id="2"),
    AIMessage(
        "The product of 317253 and 128472 plus four is 16505054788",
        name="example_assistant",
    ),
]

system = """You are bad at math but are an expert at using a calculator.

Use past tool usage as an example of how to correctly use the tools."""
few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{query}"),
    ]
)

chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools
chain.invoke("Whats 119 times 8 minus 20").tool_calls

"""
Seems like we get the correct output this time.

Here's what the [LangSmith trace](https://smith.langchain.com/public/f70550a1-585f-4c9d-a643-13148ab1616f/r) looks like.

## Next steps

- **Output parsing**: See [Ollama Tools output
    parsers](/docs/how_to/output_parser_structured)
    to learn about extracting the function calling API responses into
    various formats.
- **Structured output chains**: [Some models have constructors](/docs/how_to/structured_output) that
    handle creating a structured output chain for you.
- **Tool use**: See how to construct chains and agents that
    call the invoked tools in [these
    guides](/docs/how_to#tools).
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)
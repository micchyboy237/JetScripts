import asyncio
from jet.file.utils import save_file
from jet.search.duckduckgo import DuckDuckGoSearch, search_web
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import Image
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.tools import FunctionTool
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from io import BytesIO
# from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from autogen_ext.models.ollama import OllamaChatCompletionClient
from jet.logger import CustomLogger
from pydantic import BaseModel
from typing import Literal
import PIL
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Agents

AutoGen AgentChat provides a set of preset Agents, each with variations in how an agent might respond to messages.
All agents share the following attributes and methods:

- {py:attr}`~autogen_agentchat.agents.BaseChatAgent.name`: The unique name of the agent.
- {py:attr}`~autogen_agentchat.agents.BaseChatAgent.description`: The description of the agent in text.
- {py:attr}`~autogen_agentchat.agents.BaseChatAgent.run`: The method that runs the agent given a task as a string or a list of messages, and returns a {py:class}`~autogen_agentchat.base.TaskResult`. **Agents are expected to be stateful and this method is expected to be called with new messages, not complete history**.
- {py:attr}`~autogen_agentchat.agents.BaseChatAgent.run_stream`: Same as {py:meth}`~autogen_agentchat.agents.BaseChatAgent.run` but returns an iterator of messages that subclass {py:class}`~autogen_agentchat.messages.BaseAgentEvent` or {py:class}`~autogen_agentchat.messages.BaseChatMessage` followed by a {py:class}`~autogen_agentchat.base.TaskResult` as the last item.

See {py:mod}`autogen_agentchat.messages` for more information on AgentChat message types.

## Assistant Agent

{py:class}`~autogen_agentchat.agents.AssistantAgent` is a built-in agent that
uses a language model and has the ability to use tools.

```{warning}
{py:class}`~autogen_agentchat.agents.AssistantAgent` is a "kitchen sink" agent
for prototyping and educational purpose -- it is very general.
Make sure you read the documentation and implementation to understand the design choices.
Once you fully understand the design, you may want to implement your own agent.
See [Custom Agent](../custom-agents.ipynb).
```
"""
logger.info("# Agents")


async def web_search(query: str) -> str:
    """Find information on the web"""
    return search_web(query)


model_client = OllamaChatCompletionClient(
    model="llama3.2", log_dir=f"{OUTPUT_DIR}/chats",
)
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[web_search],
    system_message="Use tools to solve tasks.",
)

"""
## Getting Result

We can use the {py:meth}`~autogen_agentchat.agents.BaseChatAgent.run` method to get the agent run on a given task.
"""
logger.info("## Getting Result")


async def run_async_code_57d6bc29():
    result = await agent.run(task="Find information on AutoGen")
    return result
result = asyncio.run(run_async_code_57d6bc29())
logger.success(format_json(result))
logger.debug(result.messages)

"""
The call to the {py:meth}`~autogen_agentchat.agents.BaseChatAgent.run` method
returns a {py:class}`~autogen_agentchat.base.TaskResult`
with the list of messages in the {py:attr}`~autogen_agentchat.base.TaskResult.messages` attribute,
which stores the agent's "thought process" as well as the final response.

```{note}
It is important to note that {py:meth}`~autogen_agentchat.agents.BaseChatAgent.run`
will update the internal state of the agent -- it will add the messages to the agent's
message history. You can also call {py:meth}`~autogen_agentchat.agents.BaseChatAgent.run`
without a task to get the agent to generate responses given its current state.
```

```{note}
Unlike in v0.2 AgentChat, the tools are executed by the same agent directly within
the same call to {py:meth}`~autogen_agentchat.agents.BaseChatAgent.run`.
By default, the agent will return the result of the tool call as the final response.
```

## Multi-Modal Input

The {py:class}`~autogen_agentchat.agents.AssistantAgent` can handle multi-modal input
by providing the input as a {py:class}`~autogen_agentchat.messages.MultiModalMessage`.
"""
logger.info("## Multi-Modal Input")


pil_image = PIL.Image.open(
    BytesIO(requests.get("https://picsum.photos/300/200").content))
img = Image(pil_image)
multi_modal_message = MultiModalMessage(
    content=["Can you describe the content of this image?", img], source="user")
save_file(img, f"{OUTPUT_DIR}/multi_modal_img_query")


async def run_async_code_f33c45f0():
    result = await agent.run(task=multi_modal_message)
    return result
result = asyncio.run(run_async_code_f33c45f0())
logger.success(format_json(result))
logger.debug(result.messages[-1].content)  # type: ignore

"""
## Streaming Messages

We can also stream each message as it is generated by the agent by using the
{py:meth}`~autogen_agentchat.agents.BaseChatAgent.run_stream` method,
and use {py:class}`~autogen_agentchat.ui.Console` to print the messages
as they appear to the console.
"""
logger.info("## Streaming Messages")


async def assistant_run_stream() -> None:
    await Console(
        agent.run_stream(task="Find information on AutoGen"),
        output_stats=True,  # Enable stats printing.
    )


async def run_async_code_b1f495c8():
    await assistant_run_stream()
asyncio.run(run_async_code_b1f495c8())

"""
The {py:meth}`~autogen_agentchat.agents.BaseChatAgent.run_stream` method
returns an asynchronous generator that yields each message generated by the agent,
followed by a {py:class}`~autogen_agentchat.base.TaskResult` as the last item.

From the messages, you can observe that the assistant agent utilized the `web_search` tool to
gather information and responded based on the search results.

## Using Tools and Workbench

Large Language Models (LLMs) are typically limited to generating text or code responses. 
However, many complex tasks benefit from the ability to use external tools that perform specific actions,
such as fetching data from APIs or databases.

To address this limitation, modern LLMs can now accept a list of available tool schemas 
(descriptions of tools and their arguments) and generate a tool call message. 
This capability is known as **Tool Calling** or **Function Calling** and 
is becoming a popular pattern in building intelligent agent-based applications.
Refer to the documentation from [MLX](https://platform.openai.com/docs/guides/function-calling) 
and [Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) for more information about tool calling in LLMs.

In AgentChat, the {py:class}`~autogen_agentchat.agents.AssistantAgent` can use tools to perform specific actions.
The `web_search` tool is one such tool that allows the assistant agent to search the web for information.
A single custom tool can be a Python function or a subclass of the {py:class}`~autogen_core.tools.BaseTool`.

On the other hand, a {py:class}`~autogen_core.tools.Workbench` is a collection of tools that share state and resources.

```{note}
For how to use model clients directly with tools and workbench, refer to the [Tools](../../core-user-guide/components/tools.ipynb)
and [Workbench](../../core-user-guide/components/workbench.ipynb) sections
in the Core User Guide.
```

By default, when {py:class}`~autogen_agentchat.agents.AssistantAgent` executes a tool,
it will return the tool's output as a string in {py:class}`~autogen_agentchat.messages.ToolCallSummaryMessage` in its response.
If your tool does not return a well-formed string in natural language, you
can add a reflection step to have the model summarize the tool's output,
by setting the `reflect_on_tool_use=True` parameter in the {py:class}`~autogen_agentchat.agents.AssistantAgent` constructor.

### Built-in Tools and Workbench

AutoGen Extension provides a set of built-in tools that can be used with the Assistant Agent.
Head over to the [API documentation](../../../reference/index.md) for all the available tools
under the `autogen_ext.tools` namespace. For example, you can find the following tools:

- {py:mod}`~autogen_ext.tools.graphrag`: Tools for using GraphRAG index.
- {py:mod}`~autogen_ext.tools.http`: Tools for making HTTP requests.
- {py:mod}`~autogen_ext.tools.langchain`: Adaptor for using LangChain tools.
- {py:mod}`~autogen_ext.tools.mcp`: Tools and workbench for using Model Chat Protocol (MCP) servers.

### Function Tool

The {py:class}`~autogen_agentchat.agents.AssistantAgent` automatically
converts a Python function into a {py:class}`~autogen_core.tools.FunctionTool`
which can be used as a tool by the agent and automatically generates the tool schema
from the function signature and docstring.

The `web_search_func` tool is an example of a function tool.
The schema is automatically generated.
"""
logger.info("## Using Tools and Workbench")


async def web_search_func(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."


web_search_function_tool = FunctionTool(
    web_search_func, description="Find information on the web")
web_search_function_tool.schema

"""
### Model Context Protocol (MCP) Workbench

The {py:class}`~autogen_agentchat.agents.AssistantAgent` can also use tools that are
served from a Model Context Protocol (MCP) server
using {py:func}`~autogen_ext.tools.mcp.McpWorkbench`.
"""
logger.info("### Model Context Protocol (MCP) Workbench")


fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"])


async def async_func_7():
    async with McpWorkbench(fetch_mcp_server) as workbench:  # type: ignore
        model_client = MLXAutogenChatLLMAdapter(
            model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
        fetch_agent = AssistantAgent(
            name="fetcher", model_client=model_client, workbench=workbench, reflect_on_tool_use=True
        )

        result = await fetch_agent.run(task="Summarize the content of https://en.wikipedia.org/wiki/Seattle")
        logger.success(format_json(result))
        assert isinstance(result.messages[-1], TextMessage)
        logger.debug(result.messages[-1].content)

        await model_client.close()
asyncio.run(async_func_7())

"""
### Agent as a Tool

Any {py:class}`~autogen_agentchat.agents.BaseChatAgent` can be used as a tool
by wrapping it in a {py:class}`~autogen_agentchat.tools.AgentTool`.
This allows for a dynamic, model-driven multi-agent workflow where
the agent can call other agents as tools to solve tasks.

### Parallel Tool Calls

Some models support parallel tool calls, which can be useful for tasks that require multiple tools to be called simultaneously.
By default, if the model client produces multiple tool calls, {py:class}`~autogen_agentchat.agents.AssistantAgent`
will call the tools in parallel.

You may want to disable parallel tool calls when the tools have side effects that may interfere with each other, or,
when agent behavior needs to be consistent across different models.
This should be done at the model client level.

```{important}
When using {py:class}`~autogen_agentchat.tools.AgentTool` or {py:class}`~autogen_agentchat.tools.TeamTool`,
you **must** disable parallel tool calls to avoid concurrency issues.
These tools cannot run concurrently as agents and teams maintain internal state
that would conflict with parallel execution.
```

For {py:class}`~jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter.MLXAutogenChatLLMAdapter` and {py:class}`~jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter.AzureMLXAutogenChatLLMAdapter`,
set `parallel_tool_calls=False` to disable parallel tool calls.
"""
logger.info("### Agent as a Tool")

model_client_no_parallel_tool_call = MLXAutogenChatLLMAdapter(
    model="qwen3-1.7b-4bit",
    parallel_tool_calls=False,  # type: ignore
)
agent_no_parallel_tool_call = AssistantAgent(
    name="assistant",
    model_client=model_client_no_parallel_tool_call,
    tools=[web_search],
    system_message="Use tools to solve tasks.",
)

"""
### Tool Iterations

One model call followed by one tool call or parallel tool calls
is a single tool iteration.
By default, the {py:class}`~autogen_agentchat.agents.AssistantAgent` will
execute at most one iteration.

The agent can be configured to execute multiple iterations until the model
stops generating tool calls or the maximum number of iterations is reached.
You can control the maximum number of iterations by setting the `max_tool_iterations` parameter
in the {py:class}`~autogen_agentchat.agents.AssistantAgent` constructor.
"""
logger.info("### Tool Iterations")

agent_loop = AssistantAgent(
    name="assistant_loop",
    model_client=model_client_no_parallel_tool_call,
    tools=[web_search],
    system_message="Use tools to solve tasks.",
    # At most 10 iterations of tool calls before stopping the loop.
    max_tool_iterations=10,
)

"""
## Structured Output

Structured output allows models to return structured JSON text with pre-defined schema
provided by the application. Different from JSON-mode, the schema can be provided
as a [Pydantic BaseModel](https://docs.pydantic.dev/latest/concepts/models/)
class, which can also be used to validate the output.

Once you specify the base model class in the `output_content_type` parameter
of the {py:class}`~autogen_agentchat.agents.AssistantAgent` constructor,
the agent will respond with a {py:class}`~autogen_agentchat.messages.StructuredMessage`
whose `content`'s type is the type of the base model class.

This way, you can integrate agent's response directly into your application
and use the model's output as a structured object.

```{note}
When the `output_content_type` is set, it by default requires the agent to reflect on the tool use
and return the a structured output message based on the tool call result.
You can disable this behavior by setting `reflect_on_tool_use=False` explictly.
```

Structured output is also useful for incorporating Chain-of-Thought
reasoning in the agent's responses.
See the example below for how to use structured output with the assistant agent.
"""
logger.info("## Structured Output")


class AgentResponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad", "neutral"]


model_client = MLXAutogenChatLLMAdapter(
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
agent = AssistantAgent(
    "assistant",
    model_client=model_client,
    system_message="Categorize the input as happy, sad, or neutral following the JSON format.",
    output_content_type=AgentResponse,
)


async def run_async_code_7dde39b8():
    result = await Console(agent.run_stream(task="I am happy."))
    return result
result = asyncio.run(run_async_code_7dde39b8())
logger.success(format_json(result))

assert isinstance(result.messages[-1], StructuredMessage)
assert isinstance(result.messages[-1].content, AgentResponse)
logger.debug("Thought: ", result.messages[-1].content.thoughts)
logger.debug("Response: ", result.messages[-1].content.response)


async def run_async_code_0349fda4():
    await model_client.close()
asyncio.run(run_async_code_0349fda4())

"""
## Streaming Tokens

You can stream the tokens generated by the model client by setting `model_client_stream=True`.
This will cause the agent to yield {py:class}`~autogen_agentchat.messages.ModelClientStreamingChunkEvent` messages
in {py:meth}`~autogen_agentchat.agents.BaseChatAgent.run_stream`.

The underlying model API must support streaming tokens for this to work.
Please check with your model provider to see if this is supported.
"""
logger.info("## Streaming Tokens")

model_client = MLXAutogenChatLLMAdapter(
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

streaming_assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a helpful assistant.",
    model_client_stream=True,  # Enable streaming tokens.
)

# type: ignore
async for message in streaming_assistant.run_stream(task="Name two cities in South America"):
    logger.debug(message)

"""
You can see the streaming chunks in the output above.
The chunks are generated by the model client and are yielded by the agent as they are received.
The final response, the concatenation of all the chunks, is yielded right after the last chunk.

## Using Model Context

{py:class}`~autogen_agentchat.agents.AssistantAgent` has a `model_context`
parameter that can be used to pass in a {py:class}`~autogen_core.model_context.ChatCompletionContext`
object. This allows the agent to use different model contexts, such as
{py:class}`~autogen_core.model_context.BufferedChatCompletionContext` to
limit the context sent to the model.

By default, {py:class}`~autogen_agentchat.agents.AssistantAgent` uses
the {py:class}`~autogen_core.model_context.UnboundedChatCompletionContext`
which sends the full conversation history to the model. To limit the context
to the last `n` messages, you can use the {py:class}`~autogen_core.model_context.BufferedChatCompletionContext`.
To limit the context by token count, you can use the
{py:class}`~autogen_core.model_context.TokenLimitedChatCompletionContext`.
"""
logger.info("## Using Model Context")


agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[web_search],
    system_message="Use tools to solve tasks.",
    # Only use the last 5 messages in the context.
    model_context=BufferedChatCompletionContext(buffer_size=5),
)

"""
## Other Preset Agents

The following preset agents are available:

- {py:class}`~autogen_agentchat.agents.UserProxyAgent`: An agent that takes user input returns it as responses.
- {py:class}`~autogen_agentchat.agents.CodeExecutorAgent`: An agent that can execute code.
- {py:class}`~autogen_ext.agents.openai.MLXAssistantAgent`: An agent that is backed by an MLX Assistant, with ability to use custom tools.
- {py:class}`~autogen_ext.agents.web_surfer.MultimodalWebSurfer`: A multi-modal agent that can search the web and visit web pages for information.
- {py:class}`~autogen_ext.agents.file_surfer.FileSurfer`: An agent that can search and browse local files for information.
- {py:class}`~autogen_ext.agents.video_surfer.VideoSurfer`: An agent that can watch videos for information.

## Next Step

Having explored the usage of the {py:class}`~autogen_agentchat.agents.AssistantAgent`, we can now proceed to the next section to learn about the teams feature in AgentChat.

<!-- ## CodingAssistantAgent

Generates responses (text and code) using an LLM upon receipt of a message. It takes a `system_message` argument that defines or sets the tone for how the agent's LLM should respond. 

```python

writing_assistant_agent = CodingAssistantAgent(
    name="writing_assistant_agent",
    system_message="You are a helpful assistant that solve tasks by generating text responses and code.",
    model_client=model_client,
)
`

We can explore or test the behavior of the agent by sending a message to it using the  {py:meth}`~autogen_agentchat.agents.BaseChatAgent.on_messages`  method. 

```python
result = await writing_assistant_agent.on_messages(
    messages=[
        TextMessage(content="What is the weather right now in France?", source="user"),
    ],
    cancellation_token=CancellationToken(),
)
logger.debug(result) -->
"""
logger.info("## Other Preset Agents")

logger.info("\n\n[DONE]", bright=True)

import asyncio
from jet.transformers.formatters import format_json
from autogen_core import (
AgentId,
FunctionCall,
MessageContext,
RoutedAgent,
SingleThreadedAgentRuntime,
message_handler,
)
from autogen_core import CancellationToken
from autogen_core.models import (
ChatCompletionClient,
LLMMessage,
SystemMessage,
UserMessage,
)
from autogen_core.models import AssistantMessage, FunctionExecutionResult, FunctionExecutionResultMessage, UserMessage
from autogen_core.tools import FunctionTool
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OllamaChatCompletionClient
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from dataclasses import dataclass
from jet.logger import CustomLogger
from typing import List
from typing_extensions import Annotated
import asyncio
import json
import os
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Tools

Tools are code that can be executed by an agent to perform actions. A tool
can be a simple function such as a calculator, or an API call to a third-party service
such as stock price lookup or weather forecast.
In the context of AI agents, tools are designed to be executed by agents in
response to model-generated function calls.

AutoGen provides the {py:mod}`autogen_core.tools` module with a suite of built-in
tools and utilities for creating and running custom tools.

## Built-in Tools

One of the built-in tools is the {py:class}`~autogen_ext.tools.code_execution.PythonCodeExecutionTool`,
which allows agents to execute Python code snippets.

Here is how you create the tool and use it.
"""
logger.info("# Tools")


code_executor = DockerCommandLineCodeExecutor()
async def run_async_code_e817eaa6():
    await code_executor.start()
    return 
 = asyncio.run(run_async_code_e817eaa6())
logger.success(format_json())
code_execution_tool = PythonCodeExecutionTool(code_executor)
cancellation_token = CancellationToken()

code = "logger.debug('Hello, world!')"
async def run_async_code_1f3d02b5():
    async def run_async_code_82b07b81():
        result = await code_execution_tool.run_json({"code": code}, cancellation_token)
        return result
    result = asyncio.run(run_async_code_82b07b81())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_1f3d02b5())
logger.success(format_json(result))
logger.debug(code_execution_tool.return_value_as_string(result))

"""
The {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor`
class is a built-in code executor that runs Python code snippets in a subprocess
in the command line environment of a docker container.
The {py:class}`~autogen_ext.tools.code_execution.PythonCodeExecutionTool` class wraps the code executor
and provides a simple interface to execute Python code snippets.

Examples of other built-in tools
- {py:class}`~autogen_ext.tools.graphrag.LocalSearchTool` and {py:class}`~autogen_ext.tools.graphrag.GlobalSearchTool` for using [GraphRAG](https://github.com/microsoft/graphrag).
- {py:class}`~autogen_ext.tools.mcp.mcp_server_tools` for using [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) servers as tools.
- {py:class}`~autogen_ext.tools.http.HttpTool` for making HTTP requests to REST APIs.
- {py:class}`~autogen_ext.tools.langchain.LangChainToolAdapter` for using LangChain tools.

## Custom Function Tools

A tool can also be a simple Python function that performs a specific action.
To create a custom function tool, you just need to create a Python function
and use the {py:class}`~autogen_core.tools.FunctionTool` class to wrap it.

The {py:class}`~autogen_core.tools.FunctionTool` class uses descriptions and type annotations
to inform the LLM when and how to use a given function. The description provides context
about the function’s purpose and intended use cases, while type annotations inform the LLM about
the expected parameters and return type.

For example, a simple tool to obtain the stock price of a company might look like this:
"""
logger.info("## Custom Function Tools")




async def get_stock_price(ticker: str, date: Annotated[str, "Date in YYYY/MM/DD"]) -> float:
    return random.uniform(10, 200)


stock_price_tool = FunctionTool(get_stock_price, description="Get the stock price.")

cancellation_token = CancellationToken()
async def run_async_code_628fd77c():
    async def run_async_code_72e42ab1():
        result = await stock_price_tool.run_json({"ticker": "AAPL", "date": "2021/01/01"}, cancellation_token)
        return result
    result = asyncio.run(run_async_code_72e42ab1())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_628fd77c())
logger.success(format_json(result))

logger.debug(stock_price_tool.return_value_as_string(result))

"""
## Calling Tools with Model Clients

In AutoGen, every tool is a subclass of {py:class}`~autogen_core.tools.BaseTool`,
which automatically generates the JSON schema for the tool.
For example, to get the JSON schema for the `stock_price_tool`, we can use the
{py:attr}`~autogen_core.tools.BaseTool.schema` property.
"""
logger.info("## Calling Tools with Model Clients")

stock_price_tool.schema

"""
Model clients use the JSON schema of the tools to generate tool calls.

Here is an example of how to use the {py:class}`~autogen_core.tools.FunctionTool` class
with a {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient`.
Other model client classes can be used in a similar way. See [Model Clients](./model-clients.ipynb)
for more details.
"""
logger.info("Model clients use the JSON schema of the tools to generate tool calls.")



model_client = OllamaChatCompletionClient(model="llama3.1")

user_message = UserMessage(content="What is the stock price of AAPL on 2021/01/01?", source="user")

cancellation_token = CancellationToken()
async def async_func_10():
    create_result = await model_client.create(
        messages=[user_message], tools=[stock_price_tool], cancellation_token=cancellation_token
    )
    return create_result
create_result = asyncio.run(async_func_10())
logger.success(format_json(create_result))
create_result.content

"""
What is actually going on under the hood of the call to the
{py:class}`~autogen_ext.models.openai.BaseOllamaChatCompletionClient.create`
method? The model client takes the list of tools and generates a JSON schema
for the parameters of each tool. Then, it generates a request to the model
API with the tool's JSON schema and the other messages to obtain a result.

Many models, such as Ollama's GPT-4o and Llama-3.2, are trained to produce
tool calls in the form of structured JSON strings that conform to the
JSON schema of the tool. AutoGen's model clients then parse the model's response
and extract the tool call from the JSON string.

The result is a list of {py:class}`~autogen_core.FunctionCall` objects, which can be
used to run the corresponding tools.

We use `json.loads` to parse the JSON string in the {py:class}`~autogen_core.FunctionCall.arguments`
field into a Python dictionary. The {py:meth}`~autogen_core.tools.BaseTool.run_json`
method takes the dictionary and runs the tool with the provided arguments.
"""
logger.info("What is actually going on under the hood of the call to the")

assert isinstance(create_result.content, list)
arguments = json.loads(create_result.content[0].arguments)  # type: ignore
async def run_async_code_680d56f7():
    async def run_async_code_6c3c8087():
        tool_result = await stock_price_tool.run_json(arguments, cancellation_token)
        return tool_result
    tool_result = asyncio.run(run_async_code_6c3c8087())
    logger.success(format_json(tool_result))
    return tool_result
tool_result = asyncio.run(run_async_code_680d56f7())
logger.success(format_json(tool_result))
tool_result_str = stock_price_tool.return_value_as_string(tool_result)
tool_result_str

"""
Now you can make another model client call to have the model generate a reflection
on the result of the tool execution.

The result of the tool call is wrapped in a {py:class}`~autogen_core.models.FunctionExecutionResult`
object, which contains the result of the tool execution and the ID of the tool that was called.
The model client can use this information to generate a reflection on the result of the tool execution.
"""
logger.info("Now you can make another model client call to have the model generate a reflection")

exec_result = FunctionExecutionResult(
    call_id=create_result.content[0].id,  # type: ignore
    content=tool_result_str,
    is_error=False,
    name=stock_price_tool.name,
)

messages = [
    user_message,
    AssistantMessage(content=create_result.content, source="assistant"),  # assistant message with tool call
    FunctionExecutionResultMessage(content=[exec_result]),  # function execution result message
]
async def run_async_code_9d837451():
    async def run_async_code_adb24ff0():
        create_result = await model_client.create(messages=messages, cancellation_token=cancellation_token)  # type: ignore
        return create_result
    create_result = asyncio.run(run_async_code_adb24ff0())
    logger.success(format_json(create_result))
    return create_result
create_result = asyncio.run(run_async_code_9d837451())
logger.success(format_json(create_result))
logger.debug(create_result.content)
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

"""
## Tool-Equipped Agent

Putting the model client and the tools together, you can create a tool-equipped agent
that can use tools to perform actions, and reflect on the results of those actions.

```{note}
The Core API is designed to be minimal and you need to build your own agent logic around model clients and tools.
For "pre-built" agents that can use tools, please refer to the [AgentChat API](../../agentchat-user-guide/index.md).
```
"""
logger.info("## Tool-Equipped Agent")




@dataclass
class Message:
    content: str


class ToolUseAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool_schema: List[Tool]) -> None:
        super().__init__("An agent with tools")
        self._system_messages: List[LLMMessage] = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._tools = tool_schema

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]

        async def async_func_39():
            create_result = await self._model_client.create(
                messages=session,
                tools=self._tools,
                cancellation_token=ctx.cancellation_token,
            )
            return create_result
        create_result = asyncio.run(async_func_39())
        logger.success(format_json(create_result))

        if isinstance(create_result.content, str):
            return Message(content=create_result.content)
        assert isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        )

        session.append(AssistantMessage(content=create_result.content, source="assistant"))

        async def async_func_53():
            results = await asyncio.gather(
                *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
            )
            return results
        results = asyncio.run(async_func_53())
        logger.success(format_json(results))

        session.append(FunctionExecutionResultMessage(content=results))

        async def async_func_59():
            create_result = await self._model_client.create(
                messages=session,
                cancellation_token=ctx.cancellation_token,
            )
            return create_result
        create_result = asyncio.run(async_func_59())
        logger.success(format_json(create_result))
        assert isinstance(create_result.content, str)

        return Message(content=create_result.content)

    async def _execute_tool_call(
        self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        tool = next((tool for tool in self._tools if tool.name == call.name), None)
        assert tool is not None

        try:
            arguments = json.loads(call.arguments)
            async def run_async_code_05473541():
                async def run_async_code_5d3ee7b4():
                    result = await tool.run_json(arguments, cancellation_token)
                    return result
                result = asyncio.run(run_async_code_5d3ee7b4())
                logger.success(format_json(result))
                return result
            result = asyncio.run(run_async_code_05473541())
            logger.success(format_json(result))
            return FunctionExecutionResult(
                call_id=call.id, content=tool.return_value_as_string(result), is_error=False, name=tool.name
            )
        except Exception as e:
            return FunctionExecutionResult(call_id=call.id, content=str(e), is_error=True, name=tool.name)

"""
When handling a user message, the `ToolUseAgent` class first use the model client
to generate a list of function calls to the tools, and then run the tools
and generate a reflection on the results of the tool execution.
The reflection is then returned to the user as the agent's response.

To run the agent, let's create a runtime and register the agent with the runtime.
"""
logger.info("When handling a user message, the `ToolUseAgent` class first use the model client")

model_client = OllamaChatCompletionClient(model="llama3.1")
runtime = SingleThreadedAgentRuntime()
tools: List[Tool] = [FunctionTool(get_stock_price, description="Get the stock price.")]
await ToolUseAgent.register(
    runtime,
    "tool_use_agent",
    lambda: ToolUseAgent(
        model_client=model_client,
        tool_schema=tools,
    ),
)

"""
This example uses the {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient`,
for Azure Ollama and other clients, see [Model Clients](./model-clients.ipynb).
Let's test the agent with a question about stock price.
"""
logger.info("This example uses the {py:class}`~autogen_ext.models.openai.OllamaChatCompletionClient`,")

runtime.start()
tool_use_agent = AgentId("tool_use_agent", "default")
async def run_async_code_7e95725f():
    async def run_async_code_df4fcb36():
        response = await runtime.send_message(Message("What is the stock price of NVDA on 2024/06/01?"), tool_use_agent)
        return response
    response = asyncio.run(run_async_code_df4fcb36())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_7e95725f())
logger.success(format_json(response))
logger.debug(response.content)
async def run_async_code_4aaa8dea():
    await runtime.stop()
    return 
 = asyncio.run(run_async_code_4aaa8dea())
logger.success(format_json())
async def run_async_code_0349fda4():
    await model_client.close()
    return 
 = asyncio.run(run_async_code_0349fda4())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)
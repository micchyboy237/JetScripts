import asyncio
from jet.transformers.formatters import format_json
from autogen_core import (
FunctionCall,
MessageContext,
RoutedAgent,
message_handler,
)
from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import (
AssistantMessage,
ChatCompletionClient,
FunctionExecutionResult,
FunctionExecutionResultMessage,
LLMMessage,
SystemMessage,
UserMessage,
)
from autogen_core.tools import ToolResult, Workbench
from autogen_ext.models.openai import OllamaChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams
from dataclasses import dataclass
from jet.logger import CustomLogger
from typing import List
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Workbench (and MCP)

A {py:class}`~autogen_core.tools.Workbench` provides a collection of tools that share state and resources.
Different from {py:class}`~autogen_core.tools.Tool`, which provides an interface
to a single tool, a workbench provides an interface to call different tools
and receive results as the same types.

## Using Workbench

Here is an example of how to create an agent using {py:class}`~autogen_core.tools.Workbench`.
"""
logger.info("# Workbench (and MCP)")



@dataclass
class Message:
    content: str


class WorkbenchAgent(RoutedAgent):
    def __init__(
        self, model_client: ChatCompletionClient, model_context: ChatCompletionContext, workbench: Workbench
    ) -> None:
        super().__init__("An agent with a workbench")
        self._system_messages: List[LLMMessage] = [SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._model_context = model_context
        self._workbench = workbench

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        async def run_async_code_37101ad6():
            await self._model_context.add_message(UserMessage(content=message.content, source="user"))
            return 
         = asyncio.run(run_async_code_37101ad6())
        logger.success(format_json())
        logger.debug("---------User Message-----------")
        logger.debug(message.content)

        async def async_func_43():
            create_result = await self._model_client.create(
                messages=self._system_messages + (await self._model_context.get_messages()),
                tools=(await self._workbench.list_tools()),
                cancellation_token=ctx.cancellation_token,
            )
            return create_result
        create_result = asyncio.run(async_func_43())
        logger.success(format_json(create_result))

        while isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        ):
            logger.debug("---------Function Calls-----------")
            for call in create_result.content:
                logger.debug(call)

            async def run_async_code_f5c8f2a6():
                await self._model_context.add_message(AssistantMessage(content=create_result.content, source="assistant"))
                return 
             = asyncio.run(run_async_code_f5c8f2a6())
            logger.success(format_json())

            logger.debug("---------Function Call Results-----------")
            results: List[ToolResult] = []
            for call in create_result.content:
                async def async_func_61():
                    result = await self._workbench.call_tool(
                        call.name, arguments=json.loads(call.arguments), cancellation_token=ctx.cancellation_token
                    )
                    return result
                result = asyncio.run(async_func_61())
                logger.success(format_json(result))
                results.append(result)
                logger.debug(result)

            await self._model_context.add_message(
                FunctionExecutionResultMessage(
                    content=[
                        FunctionExecutionResult(
                            call_id=call.id,
                            content=result.to_text(),
                            is_error=result.is_error,
                            name=result.name,
                        )
                        for call, result in zip(create_result.content, results, strict=False)
                    ]
                )
            )

            async def async_func_81():
                create_result = await self._model_client.create(
                    messages=self._system_messages + (await self._model_context.get_messages()),
                    tools=(await self._workbench.list_tools()),
                    cancellation_token=ctx.cancellation_token,
                )
                return create_result
            create_result = asyncio.run(async_func_81())
            logger.success(format_json(create_result))

        assert isinstance(create_result.content, str)

        logger.debug("---------Final Response-----------")
        logger.debug(create_result.content)

        async def run_async_code_7ed40722():
            await self._model_context.add_message(AssistantMessage(content=create_result.content, source="assistant"))
            return 
         = asyncio.run(run_async_code_7ed40722())
        logger.success(format_json())

        return Message(content=create_result.content)

"""
In this example, the agent calls the tools provided by the workbench
in a loop until the model returns a final answer.

## MCP Workbench

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is a protocol
for providing tools and resources
to language models. An MCP server hosts a set of tools and manages their state,
while an MCP client operates from the side of the language model and
communicates with the server to access the tools, and to provide the
language model with the context it needs to use the tools effectively.

In AutoGen, we provide {py:class}`~autogen_ext.tools.mcp.McpWorkbench`
that implements an MCP client. You can use it to create an agent that
uses tools provided by MCP servers.

## Web Browsing Agent using Playwright MCP

Here is an example of how we can use the [Playwright MCP server](https://github.com/microsoft/playwright-mcp)
and the `WorkbenchAgent` class to create a web browsing agent.

You may need to install the browser dependencies for Playwright.
"""
logger.info("## MCP Workbench")



"""
Start the Playwright MCP server in a terminal.
"""
logger.info("Start the Playwright MCP server in a terminal.")



"""
Then, create the agent using the `WorkbenchAgent` class and
{py:class}`~autogen_ext.tools.mcp.McpWorkbench` with the Playwright MCP server URL.
"""
logger.info("Then, create the agent using the `WorkbenchAgent` class and")


playwright_server_params = SseServerParams(
    url="http://localhost:8931/sse",
)

async def async_func_9():
    async with McpWorkbench(playwright_server_params) as workbench:  # type: ignore
        runtime = SingleThreadedAgentRuntime()
        
        await WorkbenchAgent.register(
            runtime=runtime,
            type="WebAgent",
            factory=lambda: WorkbenchAgent(
                model_client=OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096),
                model_context=BufferedChatCompletionContext(buffer_size=10),
                workbench=workbench,
            ),
        )
        
        runtime.start()
        
        await runtime.send_message(
            Message(content="Use Bing to find out the address of Microsoft Building 99"),
            recipient=AgentId("WebAgent", "default"),
        )
        
        await runtime.stop()
    return result

result = asyncio.run(async_func_9())
logger.success(format_json(result))

logger.info("\n\n[DONE]", bright=True)
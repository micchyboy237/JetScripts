from typing import Optional
import httpx
from httpx_sse import aconnect_sse  # Add for SSE connection
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
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams
from dataclasses import dataclass
from jet.logger import CustomLogger
from typing import List
import json
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
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
        self._system_messages: List[LLMMessage] = [
            SystemMessage(content="You are a helpful AI assistant.")]
        self._model_client = model_client
        self._model_context = model_context
        self._workbench = workbench

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Add user message to model context
        await self._model_context.add_message(UserMessage(content=message.content, source="user"))
        logger.success(format_json(message.content))
        logger.debug("---------User Message-----------")
        logger.debug(message.content)

        # Create initial model response
        create_result = await self._model_client.create(
            messages=self._system_messages + (await self._model_context.get_messages()),
            tools=(await self._workbench.list_tools()),
            cancellation_token=ctx.cancellation_token,
        )
        logger.success(format_json(create_result))

        # Loop while the model returns function calls
        while isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        ):
            logger.debug("---------Function Calls-----------")
            for call in create_result.content:
                logger.debug(str(call))

            # Add assistant message with function calls to context
            await self._model_context.add_message(
                AssistantMessage(content=create_result.content,
                                 source="assistant")
            )
            logger.success(format_json([str(call)
                           for call in create_result.content]))

            logger.debug("---------Function Call Results-----------")
            results: List[ToolResult] = []
            for call in create_result.content:
                result = await self._workbench.call_tool(
                    call.name, arguments=json.loads(call.arguments), cancellation_token=ctx.cancellation_token
                )
                logger.success(format_json(result))
                results.append(result)
                logger.debug(str(result))

            # Add function execution results to context
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

            # Get next model response
            create_result = await self._model_client.create(
                messages=self._system_messages + (await self._model_context.get_messages()),
                tools=(await self._workbench.list_tools()),
                cancellation_token=ctx.cancellation_token,
            )
            logger.success(format_json(create_result))

        assert isinstance(create_result.content, str)

        logger.debug("---------Final Response-----------")
        logger.debug(create_result.content)

        # Add final assistant message to context
        await self._model_context.add_message(
            AssistantMessage(content=create_result.content, source="assistant")
        )
        logger.success(format_json(create_result.content))

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


# New function to check server availability
async def check_server_availability(url: str, timeout: float = 10.0, retries: int = 3) -> bool:
    """Check if the Playwright MCP server is reachable using SSE connection."""
    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Attempt to connect to the SSE endpoint
                async with aconnect_sse(client, "GET", url) as sse:
                    logger.info(
                        f"Server check attempt {attempt}: Connected to SSE endpoint {url}")
                    return True  # Successful connection to SSE endpoint
        except httpx.HTTPError as e:
            logger.warning(
                f"Attempt {attempt} failed to connect to {url}: {str(e)}")
            try:
                # Fallback: Try a standard GET to the base URL for more context
                base_url = url.rsplit("/sse", 1)[0]
                response = await client.get(base_url)
                logger.info(
                    f"Server check attempt {attempt}: Status {response.status_code}, Response: {response.text}")
            except httpx.HTTPError as fallback_e:
                logger.warning(
                    f"Fallback GET to {base_url} failed: {str(fallback_e)}")
            if attempt < retries:
                logger.info(f"Retrying in 2 seconds... ({attempt}/{retries})")
                await asyncio.sleep(2)
            else:
                logger.error(
                    f"Failed to connect to Playwright MCP server at {url} after {retries} attempts: {str(e)}"
                )
                logger.error(
                    f"Please ensure the server is running (e.g., 'npx @playwright/mcp@latest --port 8931') "
                    "and the /sse endpoint is accessible."
                )
                return False
    return False


async def async_func_9():
    playwright_server_params = SseServerParams(
        url="http://localhost:8931/sse",
    )

    # Check if the server is running before proceeding
    if not await check_server_availability(playwright_server_params.url):
        logger.error(
            "Playwright MCP server is not running or inaccessible at "
            f"{playwright_server_params.url}. Please ensure the server is started "
            "(e.g., run 'npx @playwright/mcp@latest --port 8931' in a terminal)."
        )
        return

    try:
        async with McpWorkbench(playwright_server_params) as workbench:
            logger.info("McpWorkbench initialized successfully")
            runtime = SingleThreadedAgentRuntime()
            await WorkbenchAgent.register(
                runtime=runtime,
                type="WebAgent",
                factory=lambda: WorkbenchAgent(
                    model_client=OllamaChatCompletionClient(
                        model="llama3.2", host="http://localhost:11434"),
                    model_context=BufferedChatCompletionContext(
                        buffer_size=10),
                    workbench=workbench,
                ),
            )
            runtime.start()
            logger.info("Sending message to WebAgent")
            await runtime.send_message(
                Message(
                    content="Use Bing to find out the address of Microsoft Building 99"),
                recipient=AgentId("WebAgent", "default"),
            )
            await runtime.stop()
    except Exception as e:
        logger.error(
            f"Failed to initialize McpWorkbench or process request: {str(e)}")
        raise

asyncio.run(async_func_9())
logger.info("\n\n[DONE]", bright=True)

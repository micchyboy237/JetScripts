import re
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
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams
from autogen_core.tools import ToolResult, Workbench
from dataclasses import dataclass
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from jet.logger import CustomLogger
from typing import List
import json
import os
import shutil
import httpx  # Add httpx for server health check

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


@dataclass
class Message:
    content: str


class WorkbenchAgent(RoutedAgent):
    def __init__(
        self, model_client: ChatCompletionClient, model_context: ChatCompletionContext, workbench: Workbench
    ) -> None:
        super().__init__("An agent with a workbench")
        self._system_messages: List[LLMMessage] = [
            SystemMessage(
                content="You are a helpful AI assistant with access to a web search tool. "
                        "For queries requiring external information (e.g., addresses, recent data), "
                        "use the provided search tool (e.g., Bing search) to retrieve accurate information. "
                        "Only provide direct answers if you are certain of the information without needing a search."
            )
        ]
        self._model_client = model_client
        self._model_context = model_context
        self._workbench = workbench

    # Update the handle_user_message method in WorkbenchAgent class
    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        await self._model_context.add_message(UserMessage(content=message.content, source="user"))
        logger.debug("---------User Message-----------")
        logger.debug(message.content)
        available_tools = await self._workbench.list_tools()
        logger.debug(f"Available tools: {format_json(available_tools)}")

        # Create initial model response with tools
        create_result = await self._model_client.create(
            messages=self._system_messages + (await self._model_context.get_messages()),
            tools=available_tools,
            tool_choice="auto",
            cancellation_token=ctx.cancellation_token,
        )
        logger.success(format_json(create_result))

        # Parse tool calls from model response
        tool_calls = self._parse_tool_calls(create_result.content)
        logger.debug("---------Function Calls-----------")
        for call in tool_calls:
            logger.debug(format_json({
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments
            }))

        # If browser_search is called, simulate it with navigate and type
        for call in tool_calls:
            if call.name == "browser_search":
                query = json.loads(call.arguments).get("query", "")
                # Navigate to Bing
                await self._workbench.call_tool(
                    "browser_navigate",
                    arguments={"url": "https://www.bing.com"},
                    cancellation_token=ctx.cancellation_token
                )
                # Type the query into the search bar
                await self._workbench.call_tool(
                    "browser_type",
                    arguments={
                        "element": "search input",
                        "ref": "input[name='q']",
                        "text": query,
                        "submit": True
                    },
                    cancellation_token=ctx.cancellation_token
                )
                # Capture snapshot to get search results
                snapshot_result = await self._workbench.call_tool(
                    "browser_snapshot",
                    arguments={},
                    cancellation_token=ctx.cancellation_token
                )
                logger.success(format_json(snapshot_result))
                await self._model_context.add_message(
                    FunctionExecutionResultMessage(
                        content=[
                            FunctionExecutionResult(
                                call_id=call.id,
                                content=snapshot_result.to_text(),
                                is_error=snapshot_result.is_error,
                                name="browser_search"
                            )
                        ]
                    )
                )
            else:
                result = await self._workbench.call_tool(
                    call.name,
                    arguments=json.loads(call.arguments),
                    cancellation_token=ctx.cancellation_token
                )
                logger.success(format_json(result))
                await self._model_context.add_message(
                    FunctionExecutionResultMessage(
                        content=[
                            FunctionExecutionResult(
                                call_id=call.id,
                                content=result.to_text(),
                                is_error=result.is_error,
                                name=result.name
                            )
                        ]
                    )
                )

        # Generate final response
        create_result = await self._model_client.create(
            messages=self._system_messages + (await self._model_context.get_messages()),
            tools=available_tools,
            tool_choice="auto",
            cancellation_token=ctx.cancellation_token,
        )
        logger.success(format_json(create_result))

        # Serialize tool calls properly
        tool_calls_serializable = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in tool_calls
        ]
        tool_calls_str = json.dumps(tool_calls_serializable)
        logger.debug("---------Final Response-----------")
        logger.debug(format_json(tool_calls_serializable))
        await self._model_context.add_message(AssistantMessage(content=create_result.content, source="assistant"))
        return Message(content=create_result.content)

    # Update the _parse_tool_calls method to handle multiple tool calls
    def _parse_tool_calls(self, content: str) -> List[FunctionCall]:
        tool_calls = []
        matches = re.findall(
            r'<tool_call>\n(.*?)\n</tool_call>', content, re.DOTALL)
        for i, match in enumerate(matches):
            try:
                tool_call_json = json.loads(match)
                tool_calls.append(FunctionCall(
                    id=f"call_{i}",
                    name=tool_call_json["name"],
                    arguments=json.dumps(tool_call_json["arguments"])
                ))
            except json.JSONDecodeError:
                logger.error(f"Failed to parse tool call JSON: {match}")
        if not tool_calls:
            raise ValueError("Failed to parse any tool calls from content")
        return tool_calls

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
                    model_client=MLXAutogenChatLLMAdapter(
                        model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
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

async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.core.workflow import (
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
    )
    from llama_index.tools.mcp import (
    get_tools_from_mcp_url,
    aget_tools_from_mcp_url,
    )
    from llama_index.tools.mcp import BasicMCPClient
    from llama_index.tools.mcp.utils import workflow_as_mcp
    from mcp.client.auth import TokenStorage
    from mcp.shared.auth import OAuthToken, OAuthClientInformationFull
    from typing import Optional
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # LlamaIndex + MCP Usage
    
    The `llama-index-tools-mcp` package provides several tools for using MCP with LlamaIndex.
    """
    logger.info("# LlamaIndex + MCP Usage")
    
    # %pip install llama-index-tools-mcp
    
    """
    ## Using Tools from an MCP Server
    
    Using the `get_tools_from_mcp_url` or `aget_tools_from_mcp_url` function, you can get a list of `FunctionTool`s from an MCP server.
    """
    logger.info("## Using Tools from an MCP Server")
    
    
    tools = await aget_tools_from_mcp_url("http://127.0.0.1:8000/mcp")
    logger.success(format_json(tools))
    logger.success(format_json(tools))
    
    """
    By default, this will use our `BasicMCPClient`, which will run a command or connect to the URL and return the tools.
    
    You can also pass in a custom `ClientSession` to use a different client.
    
    You can also specify a list of allowed tools to filter the tools that are returned.
    """
    logger.info("By default, this will use our `BasicMCPClient`, which will run a command or connect to the URL and return the tools.")
    
    
    client = BasicMCPClient("http://127.0.0.1:8000/mcp")
    
    tools = await aget_tools_from_mcp_url(
            "http://127.0.0.1:8000/mcp",
            client=client,
            allowed_tools=["tool1", "tool2"],
        )
    logger.success(format_json(tools))
    
    """
    ## Converting a Workflow to an MCP App
    
    If you have a custom `Workflow`, you can convert it to an MCP app using the `workflow_as_mcp` function.
    
    For example, let's use the following workflow that will make a string loud:
    """
    logger.info("## Converting a Workflow to an MCP App")
    
    
    
    class RunEvent(StartEvent):
        msg: str
    
    
    class InfoEvent(Event):
        msg: str
    
    
    class LoudWorkflow(Workflow):
        """Useful for converting strings to uppercase and making them louder."""
    
        @step
        def step_one(self, ctx: Context, ev: RunEvent) -> StopEvent:
            ctx.write_event_to_stream(InfoEvent(msg="Hello, world!"))
    
            return StopEvent(result=ev.msg.upper() + "!")
    
    
    workflow = LoudWorkflow()
    
    mcp = workflow_as_mcp(workflow)
    
    """
    This code will automatically generate a `FastMCP` server that will
    - Use the workflow class name as the tool name
    - Use our custom `RunEvent` as the typed inputs to the tool
    - Automatically use the SSE stream for streaming json dumps of the workflow event stream
    
    If this code was in a script called `script.py`, you could launch the MCP server with:
    
    ```bash
    mcp dev script.py
    ```
    
    Or the other commands documented in the [MCP CLI README](https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#quickstart).
    
    Note that to launch from the CLI, you may need to install the MCP CLI:
    
    ```bash
    pip install "mcp[cli]"
    ```
    
    You can further customize the `FastMCP` server by passing in additional arguments to the `workflow_as_mcp` function:
    - `workflow_name`: The name of the workflow. Defaults to the class name.
    - `workflow_description`: The description of the workflow. Defaults to the class docstring.
    - `start_event_model`: The event model to use for the start event. You can either use a custom `StartEvent` class in your workflow or pass in your own pydantic model here to define the inputs to the workflow.
    - `**fastmcp_init_kwargs`: Any extra arguments to pass to the `FastMCP()` server constructor.
    
    ## MCP Client Usage
    
    The `BasicMCPClient` provides comprehensive access to MCP server capabilities beyond just tools.
    
    ### Basic Client Operations
    """
    logger.info("## MCP Client Usage")
    
    
    http_client = BasicMCPClient("https://example.com/mcp")  # Streamable HTTP
    sse_client = BasicMCPClient("https://example.com/sse")  # Server-Sent Events
    local_client = BasicMCPClient("python", args=["server.py"])  # stdio
    
    tools = await http_client.list_tools()
    logger.success(format_json(tools))
    logger.success(format_json(tools))
    
    result = await http_client.call_tool("calculate", {"x": 5, "y": 10})
    logger.success(format_json(result))
    logger.success(format_json(result))
    
    resources = await http_client.list_resources()
    logger.success(format_json(resources))
    logger.success(format_json(resources))
    
    content, mime_type = await http_client.read_resource("config://app")
    logger.success(format_json(content, mime_type))
    logger.success(format_json(content, mime_type))
    
    prompts = await http_client.list_prompts()
    logger.success(format_json(prompts))
    logger.success(format_json(prompts))
    
    prompt_result = await http_client.get_prompt("greet", {"name": "World"})
    logger.success(format_json(prompt_result))
    logger.success(format_json(prompt_result))
    
    """
    ### OAuth Authentication
    
    The client supports OAuth 2.0 authentication for connecting to protected MCP servers.
    
    You can see the [MCP docs](https://github.com/modelcontextprotocol/python-sdk/blob/main/README.md) for full details on configuring the various aspects of OAuth for both [clients](https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#oauth-authentication-for-clients) and [servers](https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#authentication).
    """
    logger.info("### OAuth Authentication")
    
    
    client = BasicMCPClient.with_oauth(
        "https://api.example.com/mcp",
        client_name="My App",
        redirect_uris=["http://localhost:3000/callback"],
        redirect_handler=lambda url: logger.debug(f"Please visit: {url}"),
        callback_handler=lambda: (input("Enter the code: "), None),
    )
    
    tools = await client.list_tools()
    logger.success(format_json(tools))
    logger.success(format_json(tools))
    
    """
    By default, the client will use an in-memory token storage if no `token_storage` is provided. You can pass in a custom `TokenStorage` instance to use a different storage.
    
    Below is an example showing the default in-memory token storage implementation.
    """
    logger.info("By default, the client will use an in-memory token storage if no `token_storage` is provided. You can pass in a custom `TokenStorage` instance to use a different storage.")
    
    
    
    class DefaultInMemoryTokenStorage(TokenStorage):
        """
        Simple in-memory token storage implementation for OAuth authentication.
    
        This is the default storage used when none is provided to with_oauth().
        Not suitable for production use across restarts as tokens are only stored
        in memory.
        """
    
        def __init__(self):
            self._tokens: Optional[OAuthToken] = None
            self._client_info: Optional[OAuthClientInformationFull] = None
    
        async def get_tokens(self) -> Optional[OAuthToken]:
            """Get the stored OAuth tokens."""
            return self._tokens
    
        async def set_tokens(self, tokens: OAuthToken) -> None:
            """Store OAuth tokens."""
            self._tokens = tokens
    
        async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
            """Get the stored client information."""
            return self._client_info
    
        async def set_client_info(
            self, client_info: OAuthClientInformationFull
        ) -> None:
            """Store client information."""
            self._client_info = client_info
    
    
    client = BasicMCPClient.with_oauth(
        "https://api.example.com/mcp",
        client_name="My App",
        redirect_uris=["http://localhost:3000/callback"],
        redirect_handler=lambda url: logger.debug(f"Please visit: {url}"),
        callback_handler=lambda: (input("Enter the code: "), None),
        token_storage=DefaultInMemoryTokenStorage(),
    )
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())
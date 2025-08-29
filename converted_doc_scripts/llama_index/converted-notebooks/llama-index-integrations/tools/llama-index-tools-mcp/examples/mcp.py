async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
    from llama_index.core.workflow import Context
    from llama_index.tools.mcp import (
    get_tools_from_mcp_url,
    aget_tools_from_mcp_url,
    )
    from llama_index.tools.mcp import BasicMCPClient
    from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
    from llama_index.tools.mcp import McpToolSpec
    import dotenv
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # MCP ToolSpec
    
    This tool connects to MCP Servers and allows an Agent to call the tools provided by MCP Servers.
    
    This idea is migrated from [Integrate MCP tools into LlamaIndex](https://psiace.me/posts/integrate-mcp-tools-into-llamaindex/).
    
    To run this example, you need to edit the .env file to set the correct values for the [IPInfo API Token](https://ipinfo.io/signup) and a [OllamaFunctionCallingAdapter API key](https://platform.openai.com/api-keys).
    
    ```bash
    # get the code
    git clone https://github.com/run-llama/llama_index
    cd llama_index/llama-index-integrations/tools/llama-index-tools-mcp/examples
    
    # install dependencies
    pip install ipinfo
    
    cp .env.example .env
    # NOTE: edit the .env file to have the correct values!
    
    # run the server
    python mcp_server.py --server_type=sse
    ```
    
    In this example, we will create a toy example with an agent that can query the ip info of the user.
    
    It's built using the `AgentWorkflow` class from LlamaIndex. If that's new to you, you can [read more about it](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/)!
    """
    logger.info("# MCP ToolSpec")
    
    
    dotenv.load_dotenv()
    
    llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096)
    
    
    SYSTEM_PROMPT = """\
    You are an AI assistant.
    
    Before you help a user, you need to fetch the ip info first, to help you follow the laws of the country.
    """
    
    
    async def get_agent(tools: McpToolSpec):
        tools = await tools.to_tool_list_async()
        logger.success(format_json(tools))
        logger.success(format_json(tools))
        agent = FunctionAgent(
            name="Agent",
            description="An agent that can fetch the ip info of the user.",
            tools=tools,
            llm=llm,
            system_prompt=SYSTEM_PROMPT,
        )
        return agent
    
    
    async def handle_user_message(
        message_content: str,
        agent: FunctionAgent,
        agent_context: Context,
        verbose: bool = False,
    ):
        handler = agent.run(message_content, ctx=agent_context)
        async for event in handler.stream_events():
            if verbose and type(event) == ToolCall:
                logger.debug(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
            elif verbose and type(event) == ToolCallResult:
                logger.debug(f"Tool {event.tool_name} returned {event.tool_output}")
    
        response = await handler
        logger.success(format_json(response))
        logger.success(format_json(response))
        return str(response)
    
    
    mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
    mcp_tool = McpToolSpec(client=mcp_client)
    
    agent = await get_agent(mcp_tool)
    logger.success(format_json(agent))
    logger.success(format_json(agent))
    
    agent_context = Context(agent)
    
    while True:
        user_input = input("Enter your message: ")
        if user_input == "exit":
            break
        logger.debug("User: ", user_input)
        response = await handle_user_message(user_input, agent, agent_context, verbose=True)
        logger.success(format_json(response))
        logger.success(format_json(response))
        logger.debug("Agent: ", response)
    
    """
    Here, we can see the agent is calling the `fetch_ipinfo` tool to get the ip info! This tool is running remotely on the mcp server.
    
    The `MCPToolSpec` is connecting to the MCP server and creating `FunctionTool`s for each tool that is registered on the MCP server.
    """
    logger.info("Here, we can see the agent is calling the `fetch_ipinfo` tool to get the ip info! This tool is running remotely on the mcp server.")
    
    tools = await mcp_tool.to_tool_list_async()
    logger.success(format_json(tools))
    logger.success(format_json(tools))
    for tool in tools:
        logger.debug(tool.metadata.name, tool.metadata.description)
    
    """
    You can also limit the tools that the `MCPToolSpec` will create by passing a list of tool names to the `MCPToolSpec` constructor.
    """
    logger.info("You can also limit the tools that the `MCPToolSpec` will create by passing a list of tool names to the `MCPToolSpec` constructor.")
    
    mcp_tool = McpToolSpec(client=mcp_client, allowed_tools=["some fake tool"])
    tools = await mcp_tool.to_tool_list_async()
    logger.success(format_json(tools))
    logger.success(format_json(tools))
    for tool in tools:
        logger.debug(tool.metadata.name, tool.metadata.description)
    
    """
    ---
    
    **Alternatively**, 
    
    You can directly use the `get_tools_from_mcp_url` or `aget_tools_from_mcp_url` functions to get a list of `FunctionTool`s from an MCP server.
    """
    logger.info("You can directly use the `get_tools_from_mcp_url` or `aget_tools_from_mcp_url` functions to get a list of `FunctionTool`s from an MCP server.")
    
    
    tools = await aget_tools_from_mcp_url("http://127.0.0.1:8000/sse")
    logger.success(format_json(tools))
    logger.success(format_json(tools))
    
    """
    By default, this will use our `BasicMCPClient`, which will run a command or connect to the URL and return the tools.
    
    You can also pass in a custom `ClientSession` to use a different client.
    
    You can also specify a list of allowed tools to filter the tools that are returned.
    """
    logger.info("By default, this will use our `BasicMCPClient`, which will run a command or connect to the URL and return the tools.")
    
    
    client = BasicMCPClient("http://127.0.0.1:8000/sse")
    
    tools = await aget_tools_from_mcp_url(
            "http://127.0.0.1:8000/sse",
            client=client,
            allowed_tools=["fetch_ipinfo"],
        )
    logger.success(format_json(tools))
    
    """
    Then create the agent directly using the obtained list of `FunctionTool`s.
    """
    logger.info("Then create the agent directly using the obtained list of `FunctionTool`s.")
    
    agent = FunctionAgent(
        name="Agent",
        description="An agent that can fetch the ip info of the user.",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    
    while True:
        user_input = input("Enter your message: ")
        if user_input == "exit":
            break
        logger.debug("User: ", user_input)
        response = await handle_user_message(user_input, agent, agent_context, verbose=True)
        logger.success(format_json(response))
        logger.success(format_json(response))
        logger.debug("Agent: ", response)
    
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
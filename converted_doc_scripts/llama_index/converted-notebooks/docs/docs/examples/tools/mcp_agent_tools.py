async def main():
    from fastmcp import FastMCP
    from jet.logger import CustomLogger
    from llama_index.tools.notion import NotionToolSpec
    from typing import Any, Dict, List, Optional
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    # Use LlamaIndex agent tools as MCP tools
    
    We have dozens of agent tools in [LlamaHub](https://llamahub.ai/?tab=tools) and they can all be instantly used as MCP tools! This notebook shows how exactly that's done, using the [Notion Tool](https://llamahub.ai/l/tools/llama-index-tools-notion?from=tools) as an example.
    
    First we install our tool, and our MCP server:
    """
    logger.info("# Use LlamaIndex agent tools as MCP tools")
    
    # !pip install llama-index-tools-notion mcp fastmcp
    
    """
    Bring in our dependencies:
    """
    logger.info("Bring in our dependencies:")
    
    
    logger.debug("MCP fastMCP server dependencies imported successfully!")
    
    """
    Instantiate our tools using an API key:
    """
    logger.info("Instantiate our tools using an API key:")
    
    
    notion_token = "xxxx"
    tool_spec = NotionToolSpec(integration_token=notion_token)
    
    """
    Let's see what tools are available:
    """
    logger.info("Let's see what tools are available:")
    
    tools = tool_spec.to_tool_list()
    
    for i, tool in enumerate(tools):
        logger.debug(f"Tool {i+1}: {tool.metadata.name}")
    
    """
    Now we create and configure the fastMCP server, and register each tool:
    """
    logger.info("Now we create and configure the fastMCP server, and register each tool:")
    
    mcp_server = FastMCP("MCP Agent Tools Server")
    
    for tool in tools:
        mcp_server.tool(
            name=tool.metadata.name, description=tool.metadata.description
        )(tool.real_fn)
    
    """
    Now we can run our MCP server complete with our tools!
    """
    logger.info("Now we can run our MCP server complete with our tools!")
    
    await mcp_server.run_async(transport="streamable-http")
    
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
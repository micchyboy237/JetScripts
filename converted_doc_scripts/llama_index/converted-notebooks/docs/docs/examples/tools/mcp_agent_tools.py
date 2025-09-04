from typing import Any, Dict, List, Optional
import os
import shutil
import asyncio
from fastmcp import FastMCP
from jet.logger import CustomLogger
from jet.search.adapters.searxng_llama_index_tool import SearXNGSearchToolSpec
from mcp.types import Implementation, InitializeRequest, ClientCapabilities


async def main():
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    logger.info("Bring in our dependencies:")
    logger.debug(
        "MCP fastMCP server and SearXNG tool dependencies imported successfully!")

    logger.info("Instantiate our SearXNG search tool with default settings:")
    searxng_tool_spec = SearXNGSearchToolSpec()

    logger.info("Let's see what tools are available:")
    tools = searxng_tool_spec.to_tool_list()
    for i, tool in enumerate(tools):
        logger.debug(f"Tool {i+1}: {tool.metadata.name}")

    logger.info(
        "Now we create and configure the fastMCP server, and register each tool:")
    mcp_server = FastMCP(
        name="MCP Agent Tools Server",
        implementation=Implementation(
            name="MCP Agent Tools",
            version="0.1.0"
        )
    )

    # Initialize MCP server with explicit protocol version and client capabilities
    await mcp_server.initialize(
        InitializeRequest(
            method="initialize",
            params={
                "protocolVersion": "2025-06-18",
                "capabilities": ClientCapabilities(
                    tools=ToolsCapability(listChanged=True)
                ),
                "clientInfo": Implementation(
                    name="MCP Client",
                    version="0.1.0"
                )
            }
        )
    )

    # Register tools with proper schema validation
    for tool in tools:
        try:
            mcp_server.tool(
                name=tool.metadata.name,
                description=tool.metadata.description,
                input_schema=tool.metadata.inputSchema or {},
                output_schema=tool.metadata.outputSchema or {}
            )(tool.real_fn)
            logger.debug(f"Successfully registered tool: {tool.metadata.name}")
        except Exception as e:
            logger.error(
                f"Failed to register tool {tool.metadata.name}: {str(e)}")

    logger.info(
        "Example usage of SearXNG search tool to explore recent AI advancements:")
    try:
        instant_results = searxng_tool_spec.searxng_instant_search(
            query="What are the latest breakthroughs in artificial intelligence 2025?"
        )
        logger.info("Instant Search Results:")
        for i, result in enumerate(instant_results):
            logger.info(f"Answer {i+1}: {result['text'][:200]}...")

        full_results = searxng_tool_spec.searxng_full_search(
            query="recent advancements in artificial intelligence 2025",
            count=3
        )
        logger.info("\nFull Search Results:")
        for i, result in enumerate(full_results):
            logger.info(f"Result {i+1}: {result['title']} ({result['url']})")
            logger.debug(f"Content: {result['content'][:200]}...")
            if result['publishedDate']:
                logger.debug(f"Published: {result['publishedDate']}")
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")

    logger.info("Now we can run our MCP server complete with our tools!")
    try:
        await mcp_server.run_async(transport="streamable-http")
        logger.info("\n\n[DONE]", bright=True)
    except Exception as e:
        logger.error(f"MCP server failed to run: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main())

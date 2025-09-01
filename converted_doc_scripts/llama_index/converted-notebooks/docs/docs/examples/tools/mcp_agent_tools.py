async def main():
    from fastmcp import FastMCP
    from jet.logger import CustomLogger
    from jet.search.adapters.searxng_llama_index_tool import SearXNGSearchToolSpec
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
    We have dozens of agent tools in [LlamaHub](https://llamahub.ai/?tab=tools) and they can all be instantly used as MCP tools! This notebook shows how that's done, using the [SearXNG Search Tool](https://llamahub.ai/l/tools/llama-index-tools-searxng) as an example.
    First we install our tools, and our MCP server:
    """
    logger.info("Bring in our dependencies:")
    logger.debug(
        "MCP fastMCP server and SearXNG tool dependencies imported successfully!")
    """
    Instantiate our SearXNG search tool with default settings:
    """
    logger.info("Instantiate our SearXNG search tool with default settings:")
    searxng_tool_spec = SearXNGSearchToolSpec()
    """
    Let's see what tools are available:
    """
    logger.info("Let's see what tools are available:")
    tools = searxng_tool_spec.to_tool_list()
    for i, tool in enumerate(tools):
        logger.debug(f"Tool {i+1}: {tool.metadata.name}")
    """
    Now we create and configure the fastMCP server, and register each tool:
    """
    logger.info(
        "Now we create and configure the fastMCP server, and register each tool:")
    mcp_server = FastMCP("MCP Agent Tools Server")
    for tool in tools:
        mcp_server.tool(
            name=tool.metadata.name, description=tool.metadata.description
        )(tool.real_fn)
    """
    Example usage of SearXNG search tool to explore recent AI advancements:
    """
    logger.info(
        "Example usage of SearXNG search tool to explore recent AI advancements:")
    try:
        # Instant search for quick answers
        instant_results = searxng_tool_spec.searxng_instant_search(
            query="What are the latest breakthroughs in artificial intelligence 2025?"
        )
        logger.info("Instant Search Results:")
        for i, result in enumerate(instant_results):
            logger.info(f"Answer {i+1}: {result['text'][:200]}...")

        # Full search for detailed results
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

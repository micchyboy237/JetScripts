from fastmcp import FastMCP, Client

mcp = FastMCP("Playwright-Helper")

@mcp.tool
async def wait_for_login_success(page_text: str = "Welcome"):
    """
    Waits for the 'Welcome' message to appear after a login attempt.
    """
    # In a real scenario, connect to the Playwright MCP server
    # 'http://localhost:49152/sse' is an example endpoint
    async with Client("http://localhost:8931/sse") as client:
        # Call the built-in browser_wait_for tool
        result = await client.call_tool("browser_wait_for", {
            "text": page_text,
            "time": 30  # Wait up to 30 seconds
        })
        return f"Wait completed: {result}"

if __name__ == "__main__":
    mcp.run()

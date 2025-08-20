
import asyncio
import os
import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.models.model_types import LLMModelType
from jet.servers.mcp.config import MCP_SERVER_PATH, MCP_SERVER_TOOLS_TO_DISCOVER_PATH
from jet.servers.mcp.mcp_agent import query_tool_requests, query_tool_responses
from jet.servers.mcp.mcp_utils import discover_tools
from jet.transformers.formatters import format_json
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def run_discover_tools(mcp_server_path: str):
    tools = await discover_tools(mcp_server_path)
    save_file({
        "mcp_server_path": mcp_server_path,
        "count": len(tools),
        "tool_names": [tool["name"] for tool in tools],
        "tools": tools
    }, f"{OUTPUT_DIR}/tools.json")


if __name__ == "__main__":
    # asyncio.run(run_discover_tools(MCP_SERVER_PATH))
    asyncio.run(run_discover_tools(MCP_SERVER_TOOLS_TO_DISCOVER_PATH))

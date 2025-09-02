
import asyncio
import os
import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.models.model_types import LLMModelType
from jet.servers.mcp.config import MCP_SERVER_PATH
from jet.servers.mcp.mcp_agent import query_tool_requests, query_tool_responses
from jet.servers.mcp.mcp_utils import discover_tools
from jet.transformers.formatters import format_json
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

mcp_server_path = MCP_SERVER_PATH

model_path: LLMModelType = "qwen3-1.7b-4bit"


async def run_tools_req_res(llm_response_text: str, mcp_server_path: str):
    tools = await discover_tools(mcp_server_path)
    save_file(tools, f"{OUTPUT_DIR}/tools.json")

    tool_requests = await query_tool_requests(llm_response_text)
    logger.gray(f"\nTool Requests ({len(tool_requests)})")
    logger.success(format_json(tool_requests))
    save_file(tool_requests, f"{OUTPUT_DIR}/tool_requests.json")

    tool_responses = await query_tool_responses(tool_requests, tools, mcp_server_path)
    logger.gray(f"\nTool Responses ({len(tool_responses)})")
    logger.success(format_json(tool_responses))
    save_file(tool_responses, f"{OUTPUT_DIR}/tool_responses.json")

    # Store tool results for consistency with existing output
    tool_results = [response["structuredContent"]
                    for response in tool_responses]
    save_file(tool_results, f"{OUTPUT_DIR}/tool_results.json")


if __name__ == "__main__":
    llm_response_text = "<tool_call>\n{\"name\": \"navigate_to_url\", \"arguments\": {\"url\": \"https://www.iana.org\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"summarize_text\", \"arguments\": {\"text\": \"<!DOCTYPE html>\\n<html lang=\\\"en\\\">\\n<head>\\n    <meta charset=\\\"UTF-8\\\">\\n    <title>IANA - International Assigned Numbers Authority</title>\\n</head>\\n<body>\\n    <h1>IANA - International Assigned Numbers Authority</h1>\\n    <p>IANA is an international organization responsible for the global allocation of numbers and codes used in the Internet. It manages the assignment of domain names, internet protocol addresses, and other critical resources. IANA's role is to ensure that these resources are distributed fairly and efficiently across the globe.</p>\\n</body>\\n</html>\", \"max_words\": 100}}\n</tool_call>"

    asyncio.run(run_tools_req_res(llm_response_text, MCP_SERVER_PATH))

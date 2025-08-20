import asyncio
import os
from pathlib import Path

from jet.models.model_types import LLMModelType
from jet.servers.mcp.mcp_agent import chat_session

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

if __name__ == "__main__":
    mcp_server_path = str(Path(__file__).parent / "mcp_server.py")

    model_path: LLMModelType = "qwen3-1.7b-4bit"

    # sample_prompt = "Navigate to https://www.iana.org and summarize the text content in 100 words or less."
    asyncio.run(chat_session(model_path, OUTPUT_DIR))

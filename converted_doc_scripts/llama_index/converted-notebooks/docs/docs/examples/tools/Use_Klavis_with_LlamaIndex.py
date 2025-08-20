import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from klavis import Klavis
from klavis.types import McpServerName, ConnectionType
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.mcp import (
BasicMCPClient,
get_tools_from_mcp_url,
aget_tools_from_mcp_url,
)
from llama_index.tools.mcp import (
get_tools_from_mcp_url,
aget_tools_from_mcp_url,
)
from llama_index.tools.mcp import BasicMCPClient
import os
import shutil
import webbrowser


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/llamaindex/cookbook/blob/main/integrations/Klavis/Use_Klavis_with_LlamaIndex.ipynb)

# LlamaIndex + Klavis AI Integration

This tutorial demonstrates how to build AI agents using LlamaIndex's agent framework with Klavis MCP (Model Context Protocol) servers for enhanced functionality.

## Prerequisites

Before we begin, you'll need:

- **MLX API key** - Get at [openai.com](https://openai.com/)
- **Klavis API key** - Get at [klavis.ai](https://klavis.ai/)
"""
logger.info("# LlamaIndex + Klavis AI Integration")

# %pip install -qU llama-index llama-index-tools-mcp klavis


os.environ[
#     "OPENAI_API_KEY"
# ] = "YOUR_OPENAI_API_KEY"  # Replace with your actual MLX API key
os.environ[
    "KLAVIS_API_KEY"
] = "YOUR_KLAVIS_API_KEY"  # Replace with your actual Klavis API key

"""
## Case 1: YouTube AI Agent

#### Create an AI agent to summarize YouTube videos using LlamaIndex and Klavis MCP Server.

#### Step 1 - using Klavis to create youtube MCP Server
"""
logger.info("## Case 1: YouTube AI Agent")

klavis_client = Klavis(api_key=os.getenv("KLAVIS_API_KEY"))

youtube_mcp_instance = klavis_client.mcp_server.create_server_instance(
    server_name=McpServerName.YOUTUBE,
    user_id="1234",
    platform_name="Klavis",
    connection_type=ConnectionType.STREAMABLE_HTTP,
)

youtube_mcp_server_url = youtube_mcp_instance.server_url

"""
#### Step 2 - using Llamaindex to create AI Agent with the MCP Server
"""
logger.info("#### Step 2 - using Llamaindex to create AI Agent with the MCP Server")


# llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit-mini", api_key=os.getenv("OPENAI_API_KEY"))

async def async_func_8():
    youtube_tools = await aget_tools_from_mcp_url(
        youtube_mcp_server_url, client=BasicMCPClient(youtube_mcp_server_url)
    )
    return youtube_tools
youtube_tools = asyncio.run(async_func_8())
logger.success(format_json(youtube_tools))

youtube_agent = FunctionAgent(
    name="youtube_agent",
    description="Agent using MCP-based tools",
    tools=youtube_tools,
    llm=llm,
    system_prompt="You are an AI assistant that uses MCP tools.",
)

"""
#### Step 3 - Run your AI Agent to summarize your favorite video!
"""
logger.info("#### Step 3 - Run your AI Agent to summarize your favorite video!")

YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=MmiveeGxfX0&t=528s"  # pick a video you like!

async def async_func_2():
    response = await youtube_agent.run(
        f"Summarize this video: {YOUTUBE_VIDEO_URL}"
    )
    return response
response = asyncio.run(async_func_2())
logger.success(format_json(response))
logger.debug(response)

"""
✅ Nice work! You’ve successfully oursource your eyeball and summarized your favorite YouTube video!

## Case 2: Multi-Agent Workflow

#### Build a LlamaIndex AgentWorkflow that summarizes YouTube videos and sends the summary via email.

#### Step 1 - using Klavis to create YouTube and Gmail MCP Servers
"""
logger.info("## Case 2: Multi-Agent Workflow")


klavis_client = Klavis(api_key=os.getenv("KLAVIS_API_KEY"))

youtube_mcp_instance = klavis_client.mcp_server.create_server_instance(
    server_name=McpServerName.YOUTUBE,
    user_id="1234",
    platform_name="Klavis",
    connection_type=ConnectionType.STREAMABLE_HTTP,
)

gmail_mcp_instance = klavis_client.mcp_server.create_server_instance(
    server_name=McpServerName.GMAIL,
    user_id="1234",
    platform_name="Klavis",
    connection_type=ConnectionType.STREAMABLE_HTTP,
)

logger.debug("✅ Created YouTube and Gmail MCP instances")

webbrowser.open(gmail_mcp_instance.oauth_url)
logger.debug(
    f"🔐 Opening OAuth authorization for Gmail, if you are not redirected, please open the following URL in your browser: {gmail_mcp_instance.oauth_url}"
)

"""
#### Step 2 - using LlamaIndex to create Multi-Agent Workflow with the MCP Servers
"""
logger.info("#### Step 2 - using LlamaIndex to create Multi-Agent Workflow with the MCP Servers")


# llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit-mini", api_key=os.getenv("OPENAI_API_KEY"))

youtube_mcp_server_url = youtube_mcp_instance.server_url
gmail_mcp_server_url = gmail_mcp_instance.server_url

async def async_func_13():
    youtube_tools = await aget_tools_from_mcp_url(
        youtube_mcp_server_url, client=BasicMCPClient(youtube_mcp_server_url)
    )
    return youtube_tools
youtube_tools = asyncio.run(async_func_13())
logger.success(format_json(youtube_tools))
async def async_func_16():
    gmail_tools = await aget_tools_from_mcp_url(
        gmail_mcp_server_url, client=BasicMCPClient(gmail_mcp_server_url)
    )
    return gmail_tools
gmail_tools = asyncio.run(async_func_16())
logger.success(format_json(gmail_tools))

youtube_agent = FunctionAgent(
    name="youtube_agent",
    description="Agent that can summarize YouTube videos",
    tools=youtube_tools,
    llm=llm,
    system_prompt="You are a YouTube video summarization expert. Use MCP tools to analyze and summarize videos.",
    can_handoff_to=["gmail_agent"],
)

gmail_agent = FunctionAgent(
    name="gmail_agent",
    description="Agent that can send emails via Gmail",
    tools=gmail_tools,
    llm=llm,
    system_prompt="You are an email assistant. Use MCP tools to send emails via Gmail.",
)

workflow = AgentWorkflow(
    agents=[youtube_agent, gmail_agent],
    root_agent="youtube_agent",
)

logger.debug("🤖 Multi-agent workflow created with YouTube and Gmail agents!")

"""
#### Step 3 - run the workflow!
"""
logger.info("#### Step 3 - run the workflow!")

YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=MmiveeGxfX0&t=528s"  # pick a video you like!
EMAIL_RECIPIENT = "zihaolin@klavis.ai"  # Replace with your email

async def async_func_3():
    resp = await workflow.run(
        user_msg=f"Summarize this video {YOUTUBE_VIDEO_URL} and send it to {EMAIL_RECIPIENT}"
    )
    return resp
resp = asyncio.run(async_func_3())
logger.success(format_json(resp))
logger.debug("\n✅ Report:\n", resp.response.content)

"""
## Summary

In this tutorial, we explored how to integrate LlamaIndex with Klavis AI to build powerful AI agents using MCP (Model Context Protocol) servers. Here's what we accomplished:

### Key Takeaways:

1. **Single Agent Setup**: Created a YouTube AI agent that can summarize videos using the Klavis YouTube MCP server
2. **Multi-Agent Workflow**: Built a sophisticated workflow combining YouTube and Gmail agents to summarize videos and automatically send summaries via email
3. **MCP Integration**: Learned how to use Klavis MCP servers with LlamaIndex's agent framework for enhanced functionality

This integration opens up endless possibilities for building AI agents that can interact with various services and platforms through Klavis MCP servers. You can now create agents that work with YouTube, Gmail, GitHub, Slack, and many other services supported by Klavis.

Happy building! 🚀
"""
logger.info("## Summary")

logger.info("\n\n[DONE]", bright=True)
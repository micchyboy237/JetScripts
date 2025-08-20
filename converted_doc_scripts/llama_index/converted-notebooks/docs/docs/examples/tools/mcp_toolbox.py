import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.settings import Settings
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from toolbox_llamaindex import ToolboxClient
import asyncio
import os
import shutil


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
# Using MCP Toolbox with LlamaIndex

Integrate your databases with LlamaIndex agents using MCP Toolbox.

## Overview

[MCP Toolbox for Databases](https://github.com/googleapis/genai-toolbox) is an open source MCP server for databases. It was designed with enterprise-grade and production-quality in mind. It enables you to develop tools easier, faster, and more securely by handling the complexities such as connection pooling, authentication, and more.

Toolbox Tools can be seamlessly integrated with LlamaIndex applications. For more information on [getting started](https://googleapis.github.io/genai-toolbox/getting-started/local_quickstart/) or [configuring](https://googleapis.github.io/genai-toolbox/getting-started/configure/) MCP Toolbox, see the [documentation](https://googleapis.github.io/genai-toolbox/getting-started/introduction/).

![architecture](https://raw.githubusercontent.com/googleapis/genai-toolbox/refs/heads/main/docs/en/getting-started/introduction/architecture.png)

## Configure and deploy

Toolbox is an open source server that you deploy and manage yourself. For more
instructions on deploying and configuring, see the official Toolbox
documentation:

* [Installing the Server](https://googleapis.github.io/genai-toolbox/getting-started/introduction/#installing-the-server)
* [Configuring Toolbox](https://googleapis.github.io/genai-toolbox/getting-started/configure/)

### Install client SDK

Install the LlamaIndex compatible MCP Toolbox SDK package before getting started:
"""
logger.info("# Using MCP Toolbox with LlamaIndex")

pip install toolbox-llamaindex

"""
### Loading Toolbox Tools

Once your Toolbox server is configured and up and running, you can load tools from your server:
"""
logger.info("### Loading Toolbox Tools")


prompt = """
  You're a helpful hotel assistant. You handle hotel searching, booking and
  cancellations. When the user searches for a hotel, mention it's name, id,
  location and price tier. Always mention hotel ids while performing any
  searches. This is very important for any operations. For any bookings or
  cancellations, please provide the appropriate confirmation. Be sure to
  update checkin or checkout dates if mentioned by the user.
  Don't ask for confirmations from the user.
"""

queries = [
    "Find hotels in Basel with Basel in it's name.",
    "Can you book the Hilton Basel for me?",
    "Oh wait, this is too expensive. Please cancel it and book the Hyatt Regency instead.",
    "My check in dates would be from April 10, 2024 to April 19, 2024.",
]


async def run_application():
    llm = GoogleGenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash-001",
    )


    async def async_func_32():
        async with ToolboxClient("http://127.0.0.1:5000") as client:
            async def run_async_code_e2f0749c():
                tools = await client.aload_toolset()
                return tools
            tools = asyncio.run(run_async_code_e2f0749c())
            logger.success(format_json(tools))
            
            agent = AgentWorkflow.from_tools_or_functions(
                tools,
                llm=llm,
            )
            
            for tool in tools:
                logger.debug(tool.metadata)
            
            ctx = Context(agent)
            
            for query in queries:
                async def run_async_code_168f8397():
                    response = await agent.run(user_msg=query, ctx=ctx)
                    return response
                response = asyncio.run(run_async_code_168f8397())
                logger.success(format_json(response))
                logger.debug()
                logger.debug(f"---- {query} ----")
                logger.debug(str(response))
            
            
        return result

    result = asyncio.run(async_func_32())
    logger.success(format_json(result))
async def run_async_code_3a6caff3():
    await run_application()
    return 
 = asyncio.run(run_async_code_3a6caff3())
logger.success(format_json())

"""
### Advanced Toolbox Features

Toolbox has a variety of features to make developing Gen AI tools for databases.
For more information, read more about the following features:

* [Authenticated Parameters](https://googleapis.github.io/genai-toolbox/resources/tools/#authenticated-parameters): bind tool inputs to values from OIDC tokens automatically, making it easy to run sensitive queries without potentially leaking data
* [Authorized Invocations](https://googleapis.github.io/genai-toolbox/resources/tools/#authorized-invocations): restrict access to use a tool based on the users Auth token
* [OpenTelemetry](https://googleapis.github.io/genai-toolbox/how-to/export_telemetry/): get metrics and tracing from Toolbox with OpenTelemetry
"""
logger.info("### Advanced Toolbox Features")

logger.info("\n\n[DONE]", bright=True)
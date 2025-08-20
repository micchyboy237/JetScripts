import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.openapi.base import OpenAPIToolSpec
from llama_index.tools.requests.base import RequestsToolSpec
from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
import os
import requests
import shutil
import yaml


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


# os.environ["OPENAI_API_KEY"] = "sk-your-api-key"



f = requests.get(
    "https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/openai.com/1.2.0/openapi.yaml"
).text
open_api_spec = yaml.safe_load(f)


open_spec = OpenAPIToolSpec(open_api_spec)
open_spec = OpenAPIToolSpec(
    url="https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/openai.com/1.2.0/openapi.yaml"
)

requests_spec = RequestsToolSpec(
    {
        "api.openai.com": {
            "Authorization": "Bearer sk-your-key",
            "Content-Type": "application/json",
        }
    }
)

wrapped_tools = LoadAndSearchToolSpec.from_defaults(
    open_spec.to_tool_list()[0],
).to_tool_list()

agent = FunctionAgent(
    tools=[*wrapped_tools, *requests_spec.to_tool_list()],
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

logger.debug(
    async def run_async_code_bd041055():
        await agent.run("what is the base url for the server")
        return 
     = asyncio.run(run_async_code_bd041055())
    logger.success(format_json())
)

logger.debug(
    async def run_async_code_d6738595():
        await agent.run("what is the completions api")
        return 
     = asyncio.run(run_async_code_d6738595())
    logger.success(format_json())
)

logger.debug(
    async def run_async_code_836c431e():
        await agent.run("ask the completions api for a joke")
        return 
     = asyncio.run(run_async_code_836c431e())
    logger.success(format_json())
)

logger.info("\n\n[DONE]", bright=True)
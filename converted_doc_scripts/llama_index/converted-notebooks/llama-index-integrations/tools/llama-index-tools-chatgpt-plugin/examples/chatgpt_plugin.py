import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.chatgpt_plugin.base import ChatGPTPluginToolSpec
from llama_index.tools.requests.base import RequestsToolSpec
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

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)



# os.environ["OPENAI_API_KEY"] = "sk-your-key"



f = requests.get(
    "https://raw.githubusercontent.com/sisbell/chatgpt-plugin-store/main/manifests/today-currency-converter.oiconma.repl.co.json"
).text
manifest = yaml.safe_load(f)


requests_spec = RequestsToolSpec()
plugin_spec = ChatGPTPluginToolSpec(manifest)
plugin_spec = ChatGPTPluginToolSpec(
    manifest_url="https://raw.githubusercontent.com/sisbell/chatgpt-plugin-store/main/manifests/today-currency-converter.oiconma.repl.co.json"
)

agent = FunctionAgent(
    tools=[*plugin_spec.to_tool_list(), *requests_spec.to_tool_list()],
    llm=Openai(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
)

async def run_async_code_ba89bf24():
    logger.debug(await agent.run("Can you give me info on the OpenAPI plugin that was loaded"))
    return 
 = asyncio.run(run_async_code_ba89bf24())
logger.success(format_json())

async def run_async_code_8f7533d8():
    logger.debug(await agent.run("Can you convert 100 euros to CAD"))
    return 
 = asyncio.run(run_async_code_8f7533d8())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)
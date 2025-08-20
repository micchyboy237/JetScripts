import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.tools.azure_speech.base import AzureSpeechToolSpec
from llama_index.tools.azure_translate.base import AzureTranslateToolSpec
import os
import shutil
import urllib.request


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

# os.environ["OPENAI_API_KEY"] = "sk-proj-1234567890"



speech_tool = AzureSpeechToolSpec(speech_key="your-key", region="eastus")
translate_tool = AzureTranslateToolSpec(api_key="your-key", region="eastus")

agent = FunctionAgent(
    tools=[*speech_tool.to_tool_list(), *translate_tool.to_tool_list()],
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)
ctx = Context(agent)

async def run_async_code_1619861e():
    logger.debug(await agent.run('Say "hello world"', ctx=ctx))
    return 
 = asyncio.run(run_async_code_1619861e())
logger.success(format_json())


urllib.request.urlretrieve(
    "https://speechstudiorawgithubscenarioscdn.azureedge.net/call-center/sampledata/Call1_separated_16k_health_insurance.wav",
    f"{GENERATED_DIR}/speech.wav",
)

async def run_async_code_9f6692bd():
    logger.debug(await agent.run("transcribe and format conversation in data/speech.wav", ctx=ctx))
    return 
 = asyncio.run(run_async_code_9f6692bd())
logger.success(format_json())

async def run_async_code_5725abf3():
    logger.debug(await agent.run("translate the conversation into spanish", ctx=ctx))
    return 
 = asyncio.run(run_async_code_5725abf3())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)
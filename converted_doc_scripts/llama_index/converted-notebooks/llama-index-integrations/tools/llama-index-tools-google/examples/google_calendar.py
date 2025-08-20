import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.google_calendar.base import GoogleCalendarToolSpec
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



# os.environ["OPENAI_API_KEY"] = "sk-your-key"



tool_spec = GoogleCalendarToolSpec()

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

ctx = Context(agent)

async def run_async_code_15f3694f():
    await agent.run("what is the first thing on my calendar today", ctx=ctx)
    return 
 = asyncio.run(run_async_code_15f3694f())
logger.success(format_json())

await agent.run(
    "Please create an event for june 15th, 2023 at 5pm for 1 hour and invite"
    " adam@example.com to discuss tax laws",
    ctx=ctx,
)

logger.info("\n\n[DONE]", bright=True)
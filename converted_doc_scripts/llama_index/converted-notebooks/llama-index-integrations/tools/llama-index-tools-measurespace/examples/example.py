import asyncio
from jet.transformers.formatters import format_json
from dotenv import load_dotenv
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.measurespace import MeasureSpaceToolSpec
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



load_dotenv()

api_keys = {
    'hourly_weather': os.getenv('HOURLY_WEATHER_API_KEY'),
    'daily_weather': os.getenv('DAILY_WEATHER_API_KEY'),
    'daily_climate': os.getenv('DAILY_CLIMATE_API_KEY'),
    'air_quality': os.getenv('AIR_QUALITY_API_KEY'),
    'geocoding': os.getenv('GEOCODING_API_KEY'),
}

tool_spec = MeasureSpaceToolSpec(api_keys)

for tool in tool_spec.to_tool_list():
    logger.debug(tool.metadata.name)

tool_spec.get_daily_weather_forecast('New York')

tool_spec.get_latitude_longitude_from_location('New York')

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

logger.debug(
    async def run_async_code_c1f717dd():
        await agent.run("How's the temperature for New York in next 3 days?")
        return 
     = asyncio.run(run_async_code_c1f717dd())
    logger.success(format_json())
)
logger.debug(
    async def run_async_code_92d90ccb():
        await agent.run("What's the latitude and longitude of New York?")
        return 
     = asyncio.run(run_async_code_92d90ccb())
    logger.success(format_json())
)

logger.info("\n\n[DONE]", bright=True)
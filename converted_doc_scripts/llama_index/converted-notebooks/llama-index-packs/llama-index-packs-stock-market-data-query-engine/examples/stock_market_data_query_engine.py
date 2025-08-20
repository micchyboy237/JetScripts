from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
# Stock Market Data Query Engine

Here we showcase our `StockMarketDataQueryEnginePack`.
"""
logger.info("# Stock Market Data Query Engine")

# %pip install llama-index-llms-ollama


StockMarketDataQueryEnginePack = download_llama_pack(
    "StockMarketDataQueryEnginePack",
    "./stock_market_data_pack",
)

"""
#### Initialize Pack
"""
logger.info("#### Initialize Pack")


llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

stock_market_data_pack = StockMarketDataQueryEnginePack(
    ["MSFT", "AAPL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "CRM", "AMD", "INTC"],
    period="1mo",
    llm=llm,
)

stock_market_data_pack

modules = stock_market_data_pack.get_modules()
modules["stocks market data"][1]

"""
## Try Out Some Queries
"""
logger.info("## Try Out Some Queries")

response = stock_market_data_pack.run("What is the average closing price for MSFT?")

response = stock_market_data_pack.run(
    "What is AAPL's trading volume on the day after Christmas?"
)

logger.info("\n\n[DONE]", bright=True)
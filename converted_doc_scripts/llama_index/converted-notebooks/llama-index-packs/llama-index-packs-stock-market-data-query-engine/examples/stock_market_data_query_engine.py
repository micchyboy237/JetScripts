from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core.llama_pack import download_llama_pack
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

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


llm = OllamaFunctionCallingAdapter(model="llama3.2")

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

response = stock_market_data_pack.run(
    "What is the average closing price for MSFT?")

response = stock_market_data_pack.run(
    "What is AAPL's trading volume on the day after Christmas?"
)

logger.info("\n\n[DONE]", bright=True)

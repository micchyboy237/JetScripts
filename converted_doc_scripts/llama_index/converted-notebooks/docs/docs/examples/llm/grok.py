from jet.llm.mlx.base_like import MLXLike
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Grok 4

Grok from xAI uses an MLX-compatible API, so you can use it with the MLXLike integration class.
"""
logger.info("# Grok 4")

# !pip install llama-index-llms-ollama-like

grok_api_key = "xai-xxxxxxxx"


llm = MLXLike(
    model="grok-4-0709",
    api_base="https://api.x.ai/v1",
    api_key=grok_api_key,
    context_window=128000,
    is_chat_model=True,
    is_function_calling_model=False,
)

response = llm.complete("Hello World!")
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)
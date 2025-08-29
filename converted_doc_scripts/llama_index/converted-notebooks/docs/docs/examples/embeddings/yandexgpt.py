from jet.logger import CustomLogger
from llama_index.embeddings.yandexgpt import YandexGPTEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# YandexGPT
"""
logger.info("# YandexGPT")

# %pip install llama-index-embeddings-yandexgpt

# !pip install llama-index


yandexgpt_embedding = YandexGPTEmbedding(
    api_key="your-api-key", folder_id="your-folder-id"
)

text_embedding = yandexgpt_embedding._get_text_embeddings(
    ["This is a passage!", "This is another passage"]
)
logger.debug(text_embedding)

query_embedding = yandexgpt_embedding._get_query_embedding("Where is blue?")
logger.debug(query_embedding)

logger.info("\n\n[DONE]", bright=True)
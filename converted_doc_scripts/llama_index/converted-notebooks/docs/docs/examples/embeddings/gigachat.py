from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.gigachat import GigaChatEmbedding
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
# GigaChat
"""
logger.info("# GigaChat")

# %pip install llama-index-embeddings-gigachat

# !pip install llama-index


gigachat_embedding = GigaChatEmbedding(
    auth_data="your-auth-data",
    scope="your-scope",  # Set scope 'GIGACHAT_API_PERS' for personal use or 'GIGACHAT_API_CORP' for corporate use.
)

queries_embedding = gigachat_embedding._get_query_embeddings(
    ["This is a passage!", "This is another passage"]
)
logger.debug(queries_embedding)

text_embedding = gigachat_embedding._get_text_embedding("Where is blue?")
logger.debug(text_embedding)

logger.info("\n\n[DONE]", bright=True)
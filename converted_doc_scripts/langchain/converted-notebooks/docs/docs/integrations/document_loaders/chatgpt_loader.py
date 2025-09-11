from jet.logger import logger
from langchain_community.document_loaders.chatgpt import ChatGPTLoader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# ChatGPT Data

>[ChatGPT](https://chat.ollama.com) is an artificial intelligence (AI) chatbot developed by Ollama.


This notebook covers how to load `conversations.json` from your `ChatGPT` data export folder.

You can get your data export by email by going to: https://chat.ollama.com/ -> (Profile) - Settings -> Export data -> Confirm export.
"""
logger.info("# ChatGPT Data")


loader = ChatGPTLoader(log_file="./example_data/fake_conversations.json", num_logs=1)

loader.load()

logger.info("\n\n[DONE]", bright=True)
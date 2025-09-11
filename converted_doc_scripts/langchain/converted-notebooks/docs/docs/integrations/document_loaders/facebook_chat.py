from jet.logger import logger
from langchain_community.document_loaders import FacebookChatLoader
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
# Facebook Chat

>[Messenger](https://en.wikipedia.org/wiki/Messenger_(software)) is an American proprietary instant messaging app and platform developed by `Meta Platforms`. Originally developed as `Facebook Chat` in 2008, the company revamped its messaging service in 2010.

This notebook covers how to load data from the [Facebook Chats](https://www.facebook.com/business/help/1646890868956360) into a format that can be ingested into LangChain.
"""
logger.info("# Facebook Chat")




loader = FacebookChatLoader("example_data/facebook_chat.json")

loader.load()

logger.info("\n\n[DONE]", bright=True)
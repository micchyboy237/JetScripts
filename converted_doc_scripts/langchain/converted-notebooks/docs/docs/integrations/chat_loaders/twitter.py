from jet.logger import logger
from langchain_community.adapters.ollama import convert_message_to_dict
from langchain_core.messages import AIMessage
import json
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
# Twitter (via Apify)

This notebook shows how to load chat messages from Twitter to fine-tune on. We do this by utilizing Apify. 

First, use Apify to export tweets. An example
"""
logger.info("# Twitter (via Apify)")



with open("example_data/dataset_twitter-scraper_2023-08-23_22-13-19-740.json") as f:
    data = json.load(f)

tweets = [d["full_text"] for d in data if "t.co" not in d["full_text"]]
messages = [AIMessage(content=t) for t in tweets]
system_message = {"role": "system", "content": "write a tweet"}
data = [[system_message, convert_message_to_dict(m)] for m in messages]

logger.info("\n\n[DONE]", bright=True)
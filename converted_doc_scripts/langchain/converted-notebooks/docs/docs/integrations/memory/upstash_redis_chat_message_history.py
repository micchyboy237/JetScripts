from jet.logger import logger
from langchain_community.chat_message_histories import (
UpstashRedisChatMessageHistory,
)
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
# Upstash Redis

>[Upstash](https://upstash.com/docs/introduction) is a provider of the serverless `Redis`, `Kafka`, and `QStash` APIs.

This notebook goes over how to use `Upstash Redis` to store chat message history.
"""
logger.info("# Upstash Redis")


URL = "<UPSTASH_REDIS_REST_URL>"
TOKEN = "<UPSTASH_REDIS_REST_TOKEN>"

history = UpstashRedisChatMessageHistory(
    url=URL, token=TOKEN, ttl=10, session_id="my-test-session"
)

history.add_user_message("hello llm!")
history.add_ai_message("hello user!")

history.messages

logger.info("\n\n[DONE]", bright=True)
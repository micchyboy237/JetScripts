from jet.logger import logger
from langchain_community.chat_models import MiniMaxChat
from langchain_core.messages import HumanMessage
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
---
sidebar_label: MiniMax
---

# MiniMaxChat

[Minimax](https://api.minimax.chat) is a Chinese startup that provides LLM service for companies and individuals.

This example goes over how to use LangChain to interact with MiniMax Inference for Chat.
"""
logger.info("# MiniMaxChat")


os.environ["MINIMAX_GROUP_ID"] = "MINIMAX_GROUP_ID"
os.environ["MINIMAX_API_KEY"] = "MINIMAX_API_KEY"


chat = MiniMaxChat()

chat(
    [
        HumanMessage(
            content="Translate this sentence from English to French. I love programming."
        )
    ]
)

logger.info("\n\n[DONE]", bright=True)
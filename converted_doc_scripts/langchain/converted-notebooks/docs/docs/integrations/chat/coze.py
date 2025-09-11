from jet.logger import logger
from langchain_community.chat_models import ChatCoze
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
sidebar_label: Coze Chat
---

# Chat with Coze Bot

ChatCoze chat models API by coze.com. For more information, see [https://www.coze.com/open/docs/chat](https://www.coze.com/open/docs/chat)
"""
logger.info("# Chat with Coze Bot")


chat = ChatCoze(
    coze_api_base="YOUR_API_BASE",
    coze_bot_id="YOUR_BOT_ID",
    user="YOUR_USER_ID",
    conversation_id="YOUR_CONVERSATION_ID",
    streaming=False,
)

"""
Alternatively, you can set your API key and API base with:
"""
logger.info("Alternatively, you can set your API key and API base with:")


os.environ["COZE_API_KEY"] = "YOUR_API_KEY"
os.environ["COZE_API_BASE"] = "YOUR_API_BASE"

chat([HumanMessage(content="什么是扣子(coze)")])

"""
## Chat with Coze Streaming
"""
logger.info("## Chat with Coze Streaming")

chat = ChatCoze(
    coze_api_base="YOUR_API_BASE",
    coze_bot_id="YOUR_BOT_ID",
    user="YOUR_USER_ID",
    conversation_id="YOUR_CONVERSATION_ID",
    streaming=True,
)

chat([HumanMessage(content="什么是扣子(coze)")])

logger.info("\n\n[DONE]", bright=True)
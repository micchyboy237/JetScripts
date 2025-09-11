from jet.logger import logger
from langchain_community.chat_loaders.telegram import TelegramChatLoader
from langchain_community.document_loaders import TelegramChatApiLoader
from langchain_community.document_loaders import TelegramChatFileLoader
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
# Telegram

>[Telegram Messenger](https://web.telegram.org/a/) is a globally accessible freemium, cross-platform, encrypted, cloud-based and centralized instant messaging service. The application also provides optional end-to-end encrypted chats and video calling, VoIP, file sharing and several other features.


## Installation and Setup

See [setup instructions](/docs/integrations/document_loaders/telegram).

## Document Loader

See a [usage example](/docs/integrations/document_loaders/telegram).
"""
logger.info("# Telegram")


"""
## Chat loader

See a [usage example](/docs/integrations/chat_loaders/telegram).
"""
logger.info("## Chat loader")


logger.info("\n\n[DONE]", bright=True)
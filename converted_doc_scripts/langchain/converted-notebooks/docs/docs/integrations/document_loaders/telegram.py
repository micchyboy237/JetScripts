from jet.logger import logger
from langchain_community.document_loaders import (
TelegramChatApiLoader,
TelegramChatFileLoader,
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
# Telegram

>[Telegram Messenger](https://web.telegram.org/a/) is a globally accessible freemium, cross-platform, encrypted, cloud-based and centralized instant messaging service. The application also provides optional end-to-end encrypted chats and video calling, VoIP, file sharing and several other features.

This notebook covers how to load data from `Telegram` into a format that can be ingested into LangChain.
"""
logger.info("# Telegram")


loader = TelegramChatFileLoader("example_data/telegram.json")

loader.load()

"""
`TelegramChatApiLoader` loads data directly from any specified chat from Telegram. In order to export the data, you will need to authenticate your Telegram account. 

You can get the API_HASH and API_ID from https://my.telegram.org/auth?to=apps

chat_entity – recommended to be the [entity](https://docs.telethon.dev/en/stable/concepts/entities.html?highlight=Entity#what-is-an-entity) of a channel.
"""
logger.info("You can get the API_HASH and API_ID from https://my.telegram.org/auth?to=apps")

loader = TelegramChatApiLoader(
    chat_entity="<CHAT_URL>",  # recommended to use Entity here
    api_hash="<API HASH >",
    api_id="<API_ID>",
    username="",  # needed only for caching the session.
)

loader.load()

logger.info("\n\n[DONE]", bright=True)
from jet.logger import logger
from langchain_community.chat_loaders.imessage import IMessageChatLoader
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
# Apple

>[Apple Inc. (Wikipedia)](https://en.wikipedia.org/wiki/Apple_Inc.) is an American
> multinational corporation and technology company.
>
> [iMessage (Wikipedia)](https://en.wikipedia.org/wiki/IMessage) is an instant
> messaging service developed by Apple Inc. and launched in 2011.
> `iMessage` functions exclusively on Apple platforms.

## Installation and Setup

See [setup instructions](/docs/integrations/chat_loaders/imessage).

## Chat loader

It loads chat sessions from the `iMessage` `chat.db` `SQLite` file.

See a [usage example](/docs/integrations/chat_loaders/imessage).
"""
logger.info("# Apple")


logger.info("\n\n[DONE]", bright=True)
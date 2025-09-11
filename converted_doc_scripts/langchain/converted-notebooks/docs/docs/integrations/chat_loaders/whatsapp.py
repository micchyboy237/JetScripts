from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.chat_loaders.utils import (
map_ai_messages,
merge_chat_runs,
)
from langchain_community.chat_loaders.whatsapp import WhatsAppChatLoader
from langchain_core.chat_sessions import ChatSession
from typing import List
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
# WhatsApp

This notebook shows how to use the WhatsApp chat loader. This class helps map exported WhatsApp conversations to LangChain chat messages.

The process has three steps:
1. Export the chat conversations to computer
2. Create the `WhatsAppChatLoader` with the file path pointed to the json file or directory of JSON files
3. Call `loader.load()` (or `loader.lazy_load()`) to perform the conversion.

## 1. Create message dump

To make the export of your WhatsApp conversation(s), complete the following steps:

1. Open the target conversation
2. Click the three dots in the top right corner and select "More".
3. Then select "Export chat" and choose "Without media".

An example of the data format for each conversation is below:
"""
logger.info("# WhatsApp")

# %%writefile whatsapp_chat.txt
[8/15/23, 9:12:33 AM] Dr. Feather: ‎Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.
[8/15/23, 9:12:43 AM] Dr. Feather: I spotted a rare Hyacinth Macaw yesterday in the Amazon Rainforest. Such a magnificent creature!
‎[8/15/23, 9:12:48 AM] Dr. Feather: ‎image omitted
[8/15/23, 9:13:15 AM] Jungle Jane: That's stunning! Were you able to observe its behavior?
‎[8/15/23, 9:13:23 AM] Dr. Feather: ‎image omitted
[8/15/23, 9:14:02 AM] Dr. Feather: Yes, it seemed quite social with other macaws. They're known for their playful nature.
[8/15/23, 9:14:15 AM] Jungle Jane: How's the research going on parrot communication?
‎[8/15/23, 9:14:30 AM] Dr. Feather: ‎image omitted
[8/15/23, 9:14:50 AM] Dr. Feather: It's progressing well. We're learning so much about how they use sound and color to communicate.
[8/15/23, 9:15:10 AM] Jungle Jane: That's fascinating! Can't wait to read your paper on it.
[8/15/23, 9:15:20 AM] Dr. Feather: Thank you! I'll send you a draft soon.
[8/15/23, 9:25:16 PM] Jungle Jane: Looking forward to it! Keep up the great work.

"""
## 2. Create the Chat Loader

The WhatsAppChatLoader accepts the resulting zip file, unzipped directory, or the path to any of the chat `.txt` files therein.

Provide that as well as the user name you want to take on the role of "AI" when fine-tuning.
"""
logger.info("## 2. Create the Chat Loader")


loader = WhatsAppChatLoader(
    path="./whatsapp_chat.txt",
)

"""
## 3. Load messages

The `load()` (or `lazy_load`) methods return a list of "ChatSessions" that currently store the list of messages per loaded conversation.
"""
logger.info("## 3. Load messages")



raw_messages = loader.lazy_load()
merged_messages = merge_chat_runs(raw_messages)
messages: List[ChatSession] = list(
    map_ai_messages(merged_messages, sender="Dr. Feather")
)

"""
### Next Steps

You can then use these messages how you see fit, such as fine-tuning a model, few-shot example selection, or directly make predictions for the next message.
"""
logger.info("### Next Steps")


llm = ChatOllama(model="llama3.2")

for chunk in llm.stream(messages[0]["messages"]):
    logger.debug(chunk.content, end="", flush=True)

logger.info("\n\n[DONE]", bright=True)
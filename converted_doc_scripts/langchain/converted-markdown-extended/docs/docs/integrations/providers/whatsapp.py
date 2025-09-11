from jet.logger import logger
from langchain_community.document_loaders import WhatsAppChatLoader
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

>[WhatsApp](https://www.whatsapp.com/) (also called `WhatsApp Messenger`) is a freeware, cross-platform, centralized instant messaging (IM) and voice-over-IP (VoIP) service. It allows users to send text and voice messages, make voice and video calls, and share images, documents, user locations, and other content.


## Installation and Setup

There isn't any special setup for it.



## Document Loader

See a [usage example](/docs/integrations/document_loaders/whatsapp_chat).
"""
logger.info("# WhatsApp")


logger.info("\n\n[DONE]", bright=True)
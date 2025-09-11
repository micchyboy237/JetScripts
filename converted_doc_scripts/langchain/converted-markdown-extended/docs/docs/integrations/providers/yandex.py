from jet.logger import logger
from langchain_community.chat_models import ChatYandexGPT
from langchain_community.document_loaders import YandexSTTParser
from langchain_community.embeddings import YandexGPTEmbeddings
from langchain_community.llms import YandexGPT
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
# Yandex

All functionality related to Yandex Cloud

>[Yandex Cloud](https://cloud.yandex.com/en/) is a public cloud platform.

## Installation and Setup

Yandex Cloud SDK can be installed via pip from PyPI:
"""
logger.info("# Yandex")

pip install yandexcloud

"""
## LLMs

### YandexGPT

See a [usage example](/docs/integrations/llms/yandex).
"""
logger.info("## LLMs")


"""
## Chat models

### YandexGPT

See a [usage example](/docs/integrations/chat/yandex).
"""
logger.info("## Chat models")


"""
## Embedding models

### YandexGPT

See a [usage example](/docs/integrations/text_embedding/yandex).
"""
logger.info("## Embedding models")


"""
## Parser

### YandexSTTParser

It transcribes and parses audio files.

`YandexSTTParser` is similar to the `OllamaWhisperParser`.
See a [usage example with OllamaWhisperParser](/docs/integrations/document_loaders/youtube_audio).
"""
logger.info("## Parser")


logger.info("\n\n[DONE]", bright=True)
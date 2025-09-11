from jet.logger import logger
from langchain_community.chat_models.edenai import ChatEdenAI
from langchain_community.embeddings.edenai import EdenAiEmbeddings
from langchain_community.llms import EdenAI
from langchain_community.tools.edenai import (
EdenAiExplicitImageTool,
EdenAiObjectDetectionTool,
EdenAiParsingIDTool,
EdenAiParsingInvoiceTool,
EdenAiSpeechToTextTool,
EdenAiTextModerationTool,
EdenAiTextToSpeechTool,
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
# Eden AI

>[Eden AI](https://docs.edenai.co/docs/getting-started-with-eden-ai) user interface (UI)
> is designed for handling the AI projects. With `Eden AI Portal`,
> you can perform no-code AI using the best engines for the market.


## Installation and Setup

Accessing the Eden AI API requires an API key, which you can get by
[creating an account](https://app.edenai.run/user/register) and
heading [here](https://app.edenai.run/admin/account/settings).

## LLMs

See a [usage example](/docs/integrations/llms/edenai).
"""
logger.info("# Eden AI")


"""
## Chat models

See a [usage example](/docs/integrations/chat/edenai).
"""
logger.info("## Chat models")


"""
## Embedding models

See a [usage example](/docs/integrations/text_embedding/edenai).
"""
logger.info("## Embedding models")


"""
## Tools

Eden AI provides a list of tools that grants your Agent the ability to do multiple tasks, such as:
* speech to text
* text to speech
* text explicit content detection
* image explicit content detection
* object detection
* OCR invoice parsing
* OCR ID parsing

See a [usage example](/docs/integrations/tools/edenai_tools).
"""
logger.info("## Tools")


logger.info("\n\n[DONE]", bright=True)
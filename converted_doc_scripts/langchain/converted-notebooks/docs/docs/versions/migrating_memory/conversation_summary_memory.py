from jet.logger import logger
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
# Migrating off ConversationSummaryMemory or ConversationSummaryBufferMemory

Follow this guide if you're trying to migrate off one of the old memory classes listed below:


| Memory Type                          | Description                                                                                                                                          |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ConversationSummaryMemory`           | Continually summarizes the conversation history. The summary is updated after each conversation turn. The abstraction returns the summary of the conversation history. |
| `ConversationSummaryBufferMemory`     | Provides a running summary of the conversation together with the most recent messages in the conversation under the constraint that the total number of tokens in the conversation does not exceed a certain limit. |

Please follow the following [how-to guide on summarization](https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/) in LangGraph. 

This guide shows how to maintain a running summary of the conversation while discarding older messages, ensuring they aren't re-processed during later turns.
"""
logger.info("# Migrating off ConversationSummaryMemory or ConversationSummaryBufferMemory")

logger.info("\n\n[DONE]", bright=True)
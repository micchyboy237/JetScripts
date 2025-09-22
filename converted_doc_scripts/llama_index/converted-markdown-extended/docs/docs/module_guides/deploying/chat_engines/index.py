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
# Chat Engine

## Concept

Chat engine is a high-level interface for having a conversation with your data
(multiple back-and-forth instead of a single question & answer).
Think ChatGPT, but augmented with your knowledge base.

Conceptually, it is a **stateful** analogy of a [Query Engine](/python/framework/module_guides/deploying/query_engine).
By keeping track of the conversation history, it can answer questions with past context in mind.

<Aside type="tip">
If you want to ask standalone question over your data (i.e. without keeping track of conversation history), use [Query Engine](/python/framework/module_guides/deploying/query_engine) instead.
</Aside>

## Usage Pattern

Get started with:
"""
logger.info("# Chat Engine")

chat_engine = index.as_chat_engine()
response = chat_engine.chat("Tell me a joke.")

"""
To stream response:
"""
logger.info("To stream response:")

chat_engine = index.as_chat_engine()
streaming_response = chat_engine.stream_chat("Tell me a joke.")
for token in streaming_response.response_gen:
    logger.debug(token, end="")

"""
More details in the complete [usage pattern guide](/python/framework/module_guides/deploying/chat_engines/usage_pattern).

## Modules

In our [modules section](/python/framework/module_guides/deploying/chat_engines/modules), you can find corresponding tutorials to see the available chat engines in action.
"""
logger.info("## Modules")

logger.info("\n\n[DONE]", bright=True)
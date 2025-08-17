from embedchain.store.assistants import AIAssistant
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: 'Open-Source AI Assistant'
---

Embedchain also provides support for creating Open-Source AI Assistants (similar to [MLX Assistants API](https://platform.openai.com/docs/assistants/overview)) which allows you to build AI assistants within your own applications using any LLM (MLX or otherwise). An Assistant has instructions and can leverage models, tools, and knowledge to respond to user queries.

At a high level, the Open-Source AI Assistants API has the following flow:

1. Create an AI Assistant by picking a model
2. Create a Thread when a user starts a conversation
3. Add Messages to the Thread as the user ask questions
4. Run the Assistant on the Thread to trigger responses. This automatically calls the relevant tools.

Creating an Open-Source AI Assistant is a simple 3 step process.

## Step 1: Instantiate AI Assistant
"""
logger.info("## Step 1: Instantiate AI Assistant")


assistant = AIAssistant(
    name="My Assistant",
    data_sources=[{"source": "https://www.youtube.com/watch?v=U9mJuUkhUzk"}])

"""
If you want to use the existing assistant, you can do something like this:
"""
logger.info("If you want to use the existing assistant, you can do something like this:")

assistant = AIAssistant(assistant_id="asst_xxx")

assistant = AIAssistant(assistant_id="asst_xxx", thread_id="thread_xxx")

"""
## Step-2: Add data to thread

You can add any custom data source that is supported by Embedchain. Else, you can directly pass the file path on your local system and Embedchain propagates it to MLX Assistant.
"""
logger.info("## Step-2: Add data to thread")

assistant.add("/path/to/file.pdf")
assistant.add("https://www.youtube.com/watch?v=U9mJuUkhUzk")
assistant.add("https://openai.com/blog/new-models-and-developer-products-announced-at-devday")

"""
## Step-3: Chat with your AI Assistant
"""
logger.info("## Step-3: Chat with your AI Assistant")

assistant.chat("How much MLX credits were offered to attendees during MLX DevDay?")

logger.info("\n\n[DONE]", bright=True)
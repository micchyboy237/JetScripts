from jet.logger import CustomLogger
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.prompts.system import IRS_TAX_CHATBOT
from llama_index.core.prompts.system import MARKETING_WRITING_ASSISTANT
from llama_index.core.prompts.system import SHAKESPEARE_WRITING_ASSISTANT
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/chat_engine/chat_engine_personality.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Chat Engine with a Personality ✨

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Chat Engine with a Personality ✨")

# !pip install llama-index

"""
## Default
"""
logger.info("## Default")


chat_engine = SimpleChatEngine.from_defaults()
response = chat_engine.chat(
    "Say something profound and romantic about fourth of July"
)
logger.debug(response)

"""
## Shakespeare
"""
logger.info("## Shakespeare")


chat_engine = SimpleChatEngine.from_defaults(
    system_prompt=SHAKESPEARE_WRITING_ASSISTANT
)
response = chat_engine.chat(
    "Say something profound and romantic about fourth of July"
)
logger.debug(response)

"""
## Marketing
"""
logger.info("## Marketing")


chat_engine = SimpleChatEngine.from_defaults(
    system_prompt=MARKETING_WRITING_ASSISTANT
)
response = chat_engine.chat(
    "Say something profound and romantic about fourth of July"
)
logger.debug(response)

"""
## IRS Tax
"""
logger.info("## IRS Tax")


chat_engine = SimpleChatEngine.from_defaults(system_prompt=IRS_TAX_CHATBOT)
response = chat_engine.chat(
    "Say something profound and romantic about fourth of July"
)
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
from jet.logger import logger
from langchain_community.chat_models import ChatEverlyAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
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
---
sidebar_label: EverlyAI
---

# ChatEverlyAI

>[EverlyAI](https://everlyai.xyz) allows you to run your ML models at scale in the cloud. It also provides API access to [several LLM models](https://everlyai.xyz).

This notebook demonstrates the use of `langchain.chat_models.ChatEverlyAI` for [EverlyAI Hosted Endpoints](https://everlyai.xyz/).

* Set `EVERLYAI_API_KEY` environment variable
* or use the `everlyai_api_key` keyword argument
"""
logger.info("# ChatEverlyAI")

# %pip install --upgrade --quiet  langchain-ollama

# from getpass import getpass

if "EVERLYAI_API_KEY" not in os.environ:
#     os.environ["EVERLYAI_API_KEY"] = getpass()

"""
# Let's try out LLAMA model offered on EverlyAI Hosted Endpoints
"""
logger.info("# Let's try out LLAMA model offered on EverlyAI Hosted Endpoints")


messages = [
    SystemMessage(content="You are a helpful AI that shares everything you know."),
    HumanMessage(
        content="Tell me technical facts about yourself. Are you a transformer model? How many billions of parameters do you have?"
    ),
]

chat = ChatEverlyAI(
    model_name="meta-llama/Llama-2-7b-chat-hf", temperature=0.3, max_tokens=64
)
logger.debug(chat(messages).content)

"""
# EverlyAI also supports streaming responses
"""
logger.info("# EverlyAI also supports streaming responses")


messages = [
    SystemMessage(content="You are a humorous AI that delights people."),
    HumanMessage(content="Tell me a joke?"),
]

chat = ChatEverlyAI(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    temperature=0.3,
    max_tokens=64,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
chat(messages)

"""
# Let's try a different language model on EverlyAI
"""
logger.info("# Let's try a different language model on EverlyAI")


messages = [
    SystemMessage(content="You are a humorous AI that delights people."),
    HumanMessage(content="Tell me a joke?"),
]

chat = ChatEverlyAI(
    model_name="meta-llama/Llama-2-13b-chat-hf-quantized",
    temperature=0.3,
    max_tokens=128,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
chat(messages)

logger.info("\n\n[DONE]", bright=True)
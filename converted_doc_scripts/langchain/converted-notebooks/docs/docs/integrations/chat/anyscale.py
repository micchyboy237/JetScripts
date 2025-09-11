from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_community.chat_models import ChatAnyscale
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio
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
sidebar_label: Anyscale
---

# ChatAnyscale

This notebook demonstrates the use of `langchain.chat_models.ChatAnyscale` for [Anyscale Endpoints](https://endpoints.anyscale.com/).

* Set `ANYSCALE_API_KEY` environment variable
* or use the `anyscale_api_key` keyword argument
"""
logger.info("# ChatAnyscale")

# %pip install --upgrade --quiet  langchain-ollama

# from getpass import getpass

if "ANYSCALE_API_KEY" not in os.environ:
#     os.environ["ANYSCALE_API_KEY"] = getpass()

"""
# Let's try out each model offered on Anyscale Endpoints
"""
logger.info("# Let's try out each model offered on Anyscale Endpoints")


chats = {
    model: ChatAnyscale(model_name=model, temperature=1.0)
    for model in ChatAnyscale.get_available_models()
}

logger.debug(chats.keys())

"""
# We can use async methods and other stuff supported by ChatOllama

This way, the three requests will only take as long as the longest individual request.
"""
logger.info("# We can use async methods and other stuff supported by ChatOllama")



messages = [
    SystemMessage(content="You are a helpful AI that shares everything you know."),
    HumanMessage(
        content="Tell me technical facts about yourself. Are you a transformer model? How many billions of parameters do you have?"
    ),
]


async def get_msgs():
    tasks = [chat.apredict_messages(messages) for chat in chats.values()]
    responses = await asyncio.gather(*tasks)
    logger.success(format_json(responses))
    return dict(zip(chats.keys(), responses))

# import nest_asyncio

# nest_asyncio.apply()

# %%time

response_dict = asyncio.run(get_msgs())

for model_name, response in response_dict.items():
    logger.debug(f"\t{model_name}")
    logger.debug()
    logger.debug(response.content)
    logger.debug("\n---\n")

logger.info("\n\n[DONE]", bright=True)
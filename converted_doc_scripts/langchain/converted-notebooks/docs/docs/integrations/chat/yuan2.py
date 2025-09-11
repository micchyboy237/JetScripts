from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_community.chat_models import ChatYuan2
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
ChatPromptTemplate,
)
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
sidebar_label: Yuan2.0
---

# Yuan2.0

This notebook shows how to use [YUAN2 API](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/docs/inference_server.md) in LangChain with the langchain.chat_models.ChatYuan2.

[*Yuan2.0*](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/README-EN.md) is a new generation Fundamental Large Language Model developed by IEIT System. We have published all three models, Yuan 2.0-102B, Yuan 2.0-51B, and Yuan 2.0-2B. And we provide relevant scripts for pretraining, fine-tuning, and inference services for other developers. Yuan2.0 is based on Yuan1.0, utilizing a wider range of high-quality pre training data and instruction fine-tuning datasets to enhance the model's understanding of semantics, mathematics, reasoning, code, knowledge, and other aspects.

## Getting started
### Installation
First, Yuan2.0 provided an Ollama compatible API, and we integrate ChatYuan2 into langchain chat model by using Ollama client.
Therefore, ensure the ollama package is installed in your Python environment. Run the following command:
"""
logger.info("# Yuan2.0")

# %pip install --upgrade --quiet ollama

"""
### Importing the Required Modules
After installation, import the necessary modules to your Python script:
"""
logger.info("### Importing the Required Modules")


"""
### Setting Up Your API server
Setting up your Ollama compatible API server following [yuan2 ollama api server](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/docs/Yuan2_fastchat.md).
If you deployed api server locally, you can simply set `yuan2_` or anything you want.
Just make sure, the `yuan2_api_base` is set correctly.
"""
logger.info("### Setting Up Your API server")

yuan2_
yuan2_api_base = "http://127.0.0.1:8001/v1"

"""
### Initialize the ChatYuan2 Model
Here's how to initialize the chat model:
"""
logger.info("### Initialize the ChatYuan2 Model")

chat = ChatYuan2(
    yuan2_api_base="http://127.0.0.1:8001/v1",
    temperature=1.0,
    model_name="yuan2",
    max_retries=3,
    streaming=False,
)

"""
### Basic Usage
Invoke the model with system and human messages like this:
"""
logger.info("### Basic Usage")

messages = [
    SystemMessage(content="你是一个人工智能助手。"),
    HumanMessage(content="你好，你是谁？"),
]

logger.debug(chat.invoke(messages))

"""
### Basic Usage with streaming
For continuous interaction, use the streaming feature:
"""
logger.info("### Basic Usage with streaming")


chat = ChatYuan2(
    yuan2_api_base="http://127.0.0.1:8001/v1",
    temperature=1.0,
    model_name="yuan2",
    max_retries=3,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
messages = [
    SystemMessage(content="你是个旅游小助手。"),
    HumanMessage(content="给我介绍一下北京有哪些好玩的。"),
]

chat.invoke(messages)

"""
## Advanced Features
### Usage with async calls

Invoke the model with non-blocking calls, like this:
"""
logger.info("## Advanced Features")

async def basic_agenerate():
    chat = ChatYuan2(
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
    )
    messages = [
        [
            SystemMessage(content="你是个旅游小助手。"),
            HumanMessage(content="给我介绍一下北京有哪些好玩的。"),
        ]
    ]

    result = await chat.agenerate(messages)
    logger.success(format_json(result))
    logger.debug(result)


asyncio.run(basic_agenerate())

"""
### Usage with prompt template

Invoke the model with non-blocking calls and used chat template like this:
"""
logger.info("### Usage with prompt template")

async def ainvoke_with_prompt_template():

    chat = ChatYuan2(
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个诗人，擅长写诗。"),
            ("human", "给我写首诗，主题是{theme}。"),
        ]
    )
    chain = prompt | chat
    result = await chain.ainvoke({"theme": "明月"})
    logger.success(format_json(result))
    logger.debug(f"type(result): {type(result)}; {result}")

asyncio.run(ainvoke_with_prompt_template())

"""
### Usage with async calls in streaming
For non-blocking calls with streaming output, use the astream method:
"""
logger.info("### Usage with async calls in streaming")

async def basic_astream():
    chat = ChatYuan2(
        yuan2_api_base="http://127.0.0.1:8001/v1",
        temperature=1.0,
        model_name="yuan2",
        max_retries=3,
    )
    messages = [
        SystemMessage(content="你是个旅游小助手。"),
        HumanMessage(content="给我介绍一下北京有哪些好玩的。"),
    ]
    result = chat.astream(messages)
    async for chunk in result:
        logger.debug(chunk.content, end="", flush=True)


asyncio.run(basic_astream())

logger.info("\n\n[DONE]", bright=True)
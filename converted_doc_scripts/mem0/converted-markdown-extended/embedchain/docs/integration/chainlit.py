import asyncio
from jet.transformers.formatters import format_json
from embedchain import App
from jet.logger import CustomLogger
import chainlit as cl
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
title: '⛓️ Chainlit'
description: 'Integrate with Chainlit to create LLM chat apps'
---

In this example, we will learn how to use Chainlit and Embedchain together.

![chainlit-demo](https://github.com/embedchain/embedchain/assets/73601258/d6635624-5cdb-485b-bfbd-3b7c8f18bfff)

## Setup

First, install the required packages:
"""
logger.info("## Setup")

pip install embedchain chainlit

"""
## Create a Chainlit app

Create a new file called `app.py` and add the following code:
"""
logger.info("## Create a Chainlit app")



# os.environ["OPENAI_API_KEY"] = "sk-xxx"

@cl.on_chat_start
async def on_chat_start():
    app = App.from_config(config={
        'app': {
            'config': {
                'name': 'chainlit-app'
            }
        },
        'llm': {
            'config': {
                'stream': True,
            }
        }
    })
    app.add("https://www.forbes.com/profile/elon-musk/")
    app.collect_metrics = False
    cl.user_session.set("app", app)


@cl.on_message
async def on_message(message: cl.Message):
    app = cl.user_session.get("app")
    msg = cl.Message(content="")
    async def run_async_code_074f86ad():
        for chunk in await cl.make_async(app.chat)(message.content):
        return 
     = asyncio.run(run_async_code_074f86ad())
    logger.success(format_json())
        async def run_async_code_5119e137():
            await msg.stream_token(chunk)
            return 
         = asyncio.run(run_async_code_5119e137())
        logger.success(format_json())

    async def run_async_code_2653002f():
        await msg.send()
        return 
     = asyncio.run(run_async_code_2653002f())
    logger.success(format_json())

"""
## Run the app

chainlit run app.py

## Try it out

Open the app in your browser and start chatting with it!
"""
logger.info("## Run the app")

logger.info("\n\n[DONE]", bright=True)
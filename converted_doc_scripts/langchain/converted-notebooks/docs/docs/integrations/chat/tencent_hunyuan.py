from jet.logger import logger
from langchain_community.chat_models import ChatHunyuan
from langchain_core.messages import HumanMessage
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
sidebar_label: Tencent Hunyuan
---

# Tencent Hunyuan

>[Tencent's hybrid model API](https://cloud.tencent.com/document/product/1729) (`Hunyuan API`) 
> implements dialogue communication, content generation, 
> analysis and understanding, and can be widely used in various scenarios such as intelligent 
> customer service, intelligent marketing, role playing, advertising copywriting, product description,
> script creation, resume generation, article writing, code generation, data analysis, and content
> analysis.

See [more information](https://cloud.tencent.com/document/product/1729) for more details.
"""
logger.info("# Tencent Hunyuan")


chat = ChatHunyuan(
    hunyuan_app_id=111111111,
    hunyuan_secret_id="YOUR_SECRET_ID",
    hunyuan_secret_key="YOUR_SECRET_KEY",
)

chat(
    [
        HumanMessage(
            content="You are a helpful assistant that translates English to French.Translate this sentence from English to French. I love programming."
        )
    ]
)

"""
## Using ChatHunyuan with Streaming
"""
logger.info("## Using ChatHunyuan with Streaming")

chat = ChatHunyuan(
    hunyuan_app_id="YOUR_APP_ID",
    hunyuan_secret_id="YOUR_SECRET_ID",
    hunyuan_secret_key="YOUR_SECRET_KEY",
    streaming=True,
)

chat(
    [
        HumanMessage(
            content="You are a helpful assistant that translates English to French.Translate this sentence from English to French. I love programming."
        )
    ]
)

logger.info("\n\n[DONE]", bright=True)
from jet.logger import logger
from langchain_community.chat_models import ChatSparkLLM
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
# SparkLLM Chat

SparkLLM chat models API by iFlyTek. For more information, see [iFlyTek Open Platform](https://www.xfyun.cn/).

## Basic use
"""
logger.info("# SparkLLM Chat")

"""For basic init and call"""

chat = ChatSparkLLM(
    spark_app_id="<app_id>", spark_spark_api_secret="<api_secret>"
)
message = HumanMessage(content="Hello")
chat([message])

"""
- Get SparkLLM's app_id, api_key and api_secret from [iFlyTek SparkLLM API Console](https://console.xfyun.cn/services/bm3) (for more info, see [iFlyTek SparkLLM Intro](https://xinghuo.xfyun.cn/sparkapi) ), then set environment variables `IFLYTEK_SPARK_APP_ID`, `IFLYTEK_SPARK_API_KEY` and `IFLYTEK_SPARK_API_SECRET` or pass parameters when creating `ChatSparkLLM` as the demo above.

## For ChatSparkLLM with Streaming
"""
logger.info("## For ChatSparkLLM with Streaming")

chat = ChatSparkLLM(
    spark_app_id="<app_id>",
    spark_spark_api_secret="<api_secret>",
    streaming=True,
)
for chunk in chat.stream("Hello!"):
    logger.debug(chunk.content, end="")

"""
## For v2
"""
logger.info("## For v2")

"""For basic init and call"""

chat = ChatSparkLLM(
    spark_app_id="<app_id>",
    spark_spark_api_secret="<api_secret>",
    spark_api_url="wss://spark-api.xf-yun.com/v2.1/chat",
    spark_llm_domain="generalv2",
)
message = HumanMessage(content="Hello")
chat([message])

logger.info("\n\n[DONE]", bright=True)
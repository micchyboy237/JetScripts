from jet.logger import logger
from langchain_community.llms import SparkLLM
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
# SparkLLM
[SparkLLM](https://xinghuo.xfyun.cn/spark) is a large-scale cognitive model independently developed by iFLYTEK.
It has cross-domain knowledge and language understanding ability by learning a large amount of texts, codes and images.
It can understand and perform tasks based on natural dialogue.

## Prerequisite
- Get SparkLLM's app_id, api_key and api_secret from [iFlyTek SparkLLM API Console](https://console.xfyun.cn/services/bm3) (for more info, see [iFlyTek SparkLLM Intro](https://xinghuo.xfyun.cn/sparkapi) ), then set environment variables `IFLYTEK_SPARK_APP_ID`, `IFLYTEK_SPARK_API_KEY` and `IFLYTEK_SPARK_API_SECRET` or pass parameters when creating `ChatSparkLLM` as the demo above.

## Use SparkLLM
"""
logger.info("# SparkLLM")


os.environ["IFLYTEK_SPARK_APP_ID"] = "app_id"
os.environ["IFLYTEK_SPARK_API_KEY"] = "api_key"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "api_secret"


llm = SparkLLM()

res = llm.invoke("What's your name?")
logger.debug(res)

res = llm.generate(prompts=["hello!"])
res

for res in llm.stream("foo:"):
    logger.debug(res)

logger.info("\n\n[DONE]", bright=True)
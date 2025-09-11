from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms.yuan2 import Yuan2
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
# Yuan2.0

[Yuan2.0](https://github.com/IEIT-Yuan/Yuan-2.0) is a new generation Fundamental Large Language Model developed by IEIT System. We have published all three models, Yuan 2.0-102B, Yuan 2.0-51B, and Yuan 2.0-2B. And we provide relevant scripts for pretraining, fine-tuning, and inference services for other developers. Yuan2.0 is based on Yuan1.0, utilizing a wider range of high-quality pre training data and instruction fine-tuning datasets to enhance the model's understanding of semantics, mathematics, reasoning, code, knowledge, and other aspects.

This example goes over how to use LangChain to interact with `Yuan2.0`(2B/51B/102B) Inference for text generation.

Yuan2.0 set up an inference service so user just need request the inference api to get result, which is introduced in [Yuan2.0 Inference-Server](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/docs/inference_server.md).
"""
logger.info("# Yuan2.0")


infer_api = "http://127.0.0.1:8000/yuan"


yuan_llm = Yuan2(
    infer_api=infer_api,
    max_tokens=2048,
    temp=1.0,
    top_p=0.9,
    use_history=False,
)

question = "请介绍一下中国。"

logger.debug(yuan_llm.invoke(question))

logger.info("\n\n[DONE]", bright=True)
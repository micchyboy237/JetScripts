from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_community.llms import QianfanLLMEndpoint
import os
import shutil

async def main():
    
    
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
    # Baidu Qianfan
    
    Baidu AI Cloud Qianfan Platform is a one-stop large model development and service operation platform for enterprise developers. Qianfan not only provides including the model of Wenxin Yiyan (ERNIE-Bot) and the third-party open-source models, but also provides various AI development tools and the whole set of development environment, which facilitates customers to use and develop large model applications easily.
    
    Basically, those model are split into the following type:
    
    - Embedding
    - Chat
    - Completion
    
    In this notebook, we will introduce how to use langchain with [Qianfan](https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html) mainly in `Completion` corresponding
     to the package `langchain/llms` in langchain:
    
    
    
    ## API Initialization
    
    To use the LLM services based on Baidu Qianfan, you have to initialize these parameters:
    
    You could either choose to init the AK,SK in environment variables or init params:
    
    ```base
    export QIANFAN_AK=XXX
    export QIANFAN_SK=XXX
    ```
    
    ## Current supported models:
    
    - ERNIE-Bot-turbo (default models)
    - ERNIE-Bot
    - BLOOMZ-7B
    - Llama-2-7b-chat
    - Llama-2-13b-chat
    - Llama-2-70b-chat
    - Qianfan-BLOOMZ-7B-compressed
    - Qianfan-Chinese-Llama-2-7B
    - ChatGLM2-6B-32K
    - AquilaChat-7B
    """
    logger.info("# Baidu Qianfan")
    
    # %pip install -qU langchain-community
    
    """For basic init and call"""
    
    
    os.environ["QIANFAN_AK"] = "your_ak"
    os.environ["QIANFAN_SK"] = "your_sk"
    
    llm = QianfanLLMEndpoint(streaming=True)
    res = llm.invoke("hi")
    logger.debug(res)
    
    """Test for llm generate """
    res = llm.generate(prompts=["hillo?"])
    """Test for llm aio generate"""
    
    
    async def run_aio_generate():
        resp = await llm.agenerate(prompts=["Write a 20-word article about rivers."])
        logger.success(format_json(resp))
        logger.debug(resp)
    
    
    await run_aio_generate()
    
    """Test for llm stream"""
    for res in llm.stream("write a joke."):
        logger.debug(res)
    
    """Test for llm aio stream"""
    
    
    async def run_aio_stream():
        for res in llm.stream("Write a 20-word article about mountains"):
            logger.debug(res)
    
    
    await run_aio_stream()
    
    """
    ## Use different models in Qianfan
    
    In the case you want to deploy your own model based on EB or serval open sources model, you could follow these steps:
    
    - 1. （Optional, if the model are included in the default models, skip it）Deploy your model in Qianfan Console, get your own customized deploy endpoint.
    - 2. Set up the field called `endpoint` in the initialization:
    """
    logger.info("## Use different models in Qianfan")
    
    llm = QianfanLLMEndpoint(
        streaming=True,
        model="ERNIE-Bot-turbo",
        endpoint="eb-instant",
    )
    res = llm.invoke("hi")
    
    """
    ## Model Params:
    
    For now, only `ERNIE-Bot` and `ERNIE-Bot-turbo` support model params below, we might support more models in the future.
    
    - temperature
    - top_p
    - penalty_score
    """
    logger.info("## Model Params:")
    
    res = llm.generate(
        prompts=["hi"],
        streaming=True,
        **{"top_p": 0.4, "temperature": 0.1, "penalty_score": 1},
    )
    
    for r in res:
        logger.debug(r)
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())
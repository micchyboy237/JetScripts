from jet.transformers.formatters import format_json
from jet.logger import logger
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
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
    
    In this notebook, we will introduce how to use langchain with [Qianfan](https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html) mainly in `Embedding` corresponding
     to the package `langchain/embeddings` in langchain:
    
    
    
    ## API Initialization
    
    To use the LLM services based on Baidu Qianfan, you have to initialize these parameters:
    
    You could either choose to init the AK,SK in environment variables or init params:
    
    ```base
    export QIANFAN_AK=XXX
    export QIANFAN_SK=XXX
    ```
    """
    logger.info("# Baidu Qianfan")
    
    """For basic init and call"""
    
    
    os.environ["QIANFAN_AK"] = "your_ak"
    os.environ["QIANFAN_SK"] = "your_sk"
    
    embed = QianfanEmbeddingsEndpoint(
    )
    res = embed.embed_documents(["hi", "world"])
    
    
    async def aioEmbed():
        res = await embed.aembed_query("qianfan")
        logger.success(format_json(res))
        logger.debug(res[:8])
    
    
    await aioEmbed()
    
    
    async def aioEmbedDocs():
        res = await embed.aembed_documents(["hi", "world"])
        logger.success(format_json(res))
        for r in res:
            logger.debug("", r[:8])
    
    
    await aioEmbedDocs()
    
    """
    ## Use different models in Qianfan
    
    In the case you want to deploy your own model based on Ernie Bot or third-party open sources model, you could follow these steps:
    
    - 1. （Optional, if the model are included in the default models, skip it）Deploy your model in Qianfan Console, get your own customized deploy endpoint.
    - 2. Set up the field called `endpoint` in the initialization:
    """
    logger.info("## Use different models in Qianfan")
    
    embed = QianfanEmbeddingsEndpoint(model="bge_large_zh", endpoint="bge_large_zh")
    
    res = embed.embed_documents(["hi", "world"])
    for r in res:
        logger.debug(r[:8])
    
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
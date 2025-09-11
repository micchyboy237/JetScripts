from jet.logger import logger
from langchain_community.llms import YiLLM
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
# Yi
[01.AI](https://www.lingyiwanwu.com/en), founded by Dr. Kai-Fu Lee, is a global company at the forefront of AI 2.0. They offer cutting-edge large language models, including the Yi series, which range from 6B to hundreds of billions of parameters. 01.AI also provides multimodal models, an open API platform, and open-source options like Yi-34B/9B/6B and Yi-VL.
"""
logger.info("# Yi")

# %pip install -qU langchain-community

"""
## Prerequisite
An API key is required to access Yi LLM API. Visit https://www.lingyiwanwu.com/ to get your API key. When applying for the API key, you need to specify whether it's for domestic (China) or international use.

## Use Yi LLM
"""
logger.info("## Prerequisite")


os.environ["YI_API_KEY"] = "YOUR_API_KEY"


llm = YiLLM(model="yi-large")


res = llm.invoke("What's your name?")
logger.debug(res)

res = llm.generate(
    prompts=[
        "Explain the concept of large language models.",
        "What are the potential applications of AI in healthcare?",
    ]
)
logger.debug(res)

for chunk in llm.stream("Describe the key features of the Yi language model series."):
    logger.debug(chunk, end="", flush=True)



async def run_aio_stream():
    for chunk in llm.stream(
        "Write a brief on the future of AI according to Dr. Kai-Fu Lee's vision."
    ):
        logger.debug(chunk, end="", flush=True)


asyncio.run(run_aio_stream())

llm_with_params = YiLLM(
    model="yi-large",
    temperature=0.7,
    top_p=0.9,
)

res = llm_with_params(
    "Propose an innovative AI application that could benefit society."
)
logger.debug(res)

logger.info("\n\n[DONE]", bright=True)
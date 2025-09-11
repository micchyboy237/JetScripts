from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import Aphrodite
from langchain_core.prompts import PromptTemplate
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
# Aphrodite Engine

[Aphrodite](https://github.com/PygmalionAI/aphrodite-engine) is the open-source large-scale inference engine designed to serve thousands of users on the [PygmalionAI](https://pygmalion.chat) website.

* Attention mechanism by vLLM for fast throughput and low latencies 
* Support for for many SOTA sampling methods
* Exllamav2 GPTQ kernels for better throughput at lower batch sizes

This notebooks goes over how to use a LLM with langchain and Aphrodite.

To use, you should have the `aphrodite-engine` python package installed.
"""
logger.info("# Aphrodite Engine")

# %pip install -qU langchain-community

# %pip install --upgrade --quiet  aphrodite-engine==0.4.2


llm = Aphrodite(
    model="PygmalionAI/pygmalion-2-7b",
    trust_remote_code=True,  # mandatory for hf models
    max_tokens=128,
    temperature=1.2,
    min_p=0.05,
    mirostat_mode=0,  # change to 2 to use mirostat
    mirostat_tau=5.0,
    mirostat_eta=0.1,
)

logger.debug(
    llm.invoke(
        '<|system|>Enter RP mode. You are Ayumu "Osaka" Kasuga.<|user|>Hey Osaka. Tell me about yourself.<|model|>'
    )
)

"""
## Integrate the model in an LLMChain
"""
logger.info("## Integrate the model in an LLMChain")


template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who was the US president in the year the first Pokemon game was released?"

logger.debug(llm_chain.run(question))

"""
## Distributed Inference

Aphrodite supports distributed tensor-parallel inference and serving. 

To run multi-GPU inference with the LLM class, set the `tensor_parallel_size` argument to the number of GPUs you want to use. For example, to run inference on 4 GPUs
"""
logger.info("## Distributed Inference")


llm = Aphrodite(
    model="PygmalionAI/mythalion-13b",
    tensor_parallel_size=4,
    trust_remote_code=True,  # mandatory for hf models
)

llm("What is the future of AI?")

logger.info("\n\n[DONE]", bright=True)
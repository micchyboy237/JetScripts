from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain_core.output_parsers import StrOutputParser
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
# Shale Protocol

[Shale Protocol](https://shaleprotocol.com) provides production-ready inference APIs for open LLMs. It's a Plug & Play API as it's hosted on a highly scalable GPU cloud infrastructure.

Our free tier supports up to 1K daily requests per key as we want to eliminate the barrier for anyone to start building genAI apps with LLMs.

With Shale Protocol, developers/researchers can create apps and explore the capabilities of open LLMs at no cost.

This page covers how Shale-Serve API can be incorporated with LangChain.

As of June 2023, the API supports Vicuna-13B by default. We are going to support more LLMs such as Falcon-40B in future releases.

## How to

### 1. Find the link to our Discord on https://shaleprotocol.com. Generate an API key through the "Shale Bot" on our Discord. No credit card is required and no free trials. It's a forever free tier with 1K limit per day per API key

### 2. Use https://shale.live/v1 as Ollama API drop-in replacement

For example
"""
logger.info("# Shale Protocol")


os.environ['OPENAI_API_BASE'] = "https://shale.live/v1"
# os.environ['OPENAI_API_KEY'] = "ENTER YOUR API KEY"

llm = Ollama()

template = """Question: {question}


prompt = PromptTemplate.from_template(template)


llm_chain = prompt | llm | StrOutputParser()

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.invoke(question)

logger.info("\n\n[DONE]", bright=True)
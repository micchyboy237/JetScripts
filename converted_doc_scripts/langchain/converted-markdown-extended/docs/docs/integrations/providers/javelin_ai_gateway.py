from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatJavelinAIGateway
from langchain_community.embeddings import JavelinAIGatewayEmbeddings
from langchain_community.llms import JavelinAIGateway
from langchain_core.messages import HumanMessage, SystemMessage
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
# Javelin AI Gateway

[The Javelin AI Gateway](https://www.getjavelin.io) service is a high-performance, enterprise grade API Gateway for AI applications.
It is designed to streamline the usage and access of various large language model (LLM) providers,
such as Ollama, Cohere, Ollama and custom large language models within an organization by incorporating
robust access security for all interactions with LLMs.

Javelin offers a high-level interface that simplifies the interaction with LLMs by providing a unified endpoint
to handle specific LLM related requests.

See the Javelin AI Gateway [documentation](https://docs.getjavelin.io) for more details.
[Javelin Python SDK](https://www.github.com/getjavelin/javelin-python) is an easy to use client library meant to be embedded into AI Applications

## Installation and Setup

Install `javelin_sdk` to interact with Javelin AI Gateway:
"""
logger.info("# Javelin AI Gateway")

pip install 'javelin_sdk'

"""
Set the Javelin's API key as an environment variable:
"""
logger.info("Set the Javelin's API key as an environment variable:")

export JAVELIN_API_KEY = ...

"""
## Completions Example
"""
logger.info("## Completions Example")


route_completions = "eng_dept03"

gateway = JavelinAIGateway(
    gateway_uri="http://localhost:8000",
    route=route_completions,
    model_name="text-davinci-003",
)

llmchain = LLMChain(llm=gateway, prompt=prompt)
result = llmchain.run("podcast player")

logger.debug(result)

"""
## Embeddings Example
"""
logger.info("## Embeddings Example")


embeddings = JavelinAIGatewayEmbeddings(
    gateway_uri="http://localhost:8000",
    route="embeddings",
)

logger.debug(embeddings.embed_query("hello"))
logger.debug(embeddings.embed_documents(["hello"]))

"""
## Chat Example
"""
logger.info("## Chat Example")


messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Artificial Intelligence has the power to transform humanity and make the world a better place"
    ),
]

chat = ChatJavelinAIGateway(
    gateway_uri="http://localhost:8000",
    route="mychatbot_route",
    model_name="gpt-3.5-turbo"
    params={
        "temperature": 0.1
    }
)

logger.debug(chat(messages))

logger.info("\n\n[DONE]", bright=True)

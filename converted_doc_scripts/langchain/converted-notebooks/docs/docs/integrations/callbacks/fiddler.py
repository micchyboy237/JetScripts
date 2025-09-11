from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.callbacks.fiddler_callback import FiddlerCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
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
# Fiddler

>[Fiddler](https://www.fiddler.ai/) is the pioneer in enterprise Generative and Predictive system ops, offering a unified platform that enables Data Science, MLOps, Risk, Compliance, Analytics, and other LOB teams to monitor, explain, analyze, and improve ML deployments at enterprise scale.

## 1. Installation and Setup
"""
logger.info("# Fiddler")


"""
## 2. Fiddler connection details

*Before you can add information about your model with Fiddler*

1. The URL you're using to connect to Fiddler
2. Your organization ID
3. Your authorization token

These can be found by navigating to the *Settings* page of your Fiddler environment.
"""
logger.info("## 2. Fiddler connection details")

# Your Fiddler instance URL, Make sure to include the full URL (including https://). For example: https://demo.fiddler.ai
URL = ""
ORG_NAME = ""
AUTH_TOKEN = ""  # Your Fiddler instance auth token

PROJECT_NAME = ""
MODEL_NAME = ""  # Model name in Fiddler

"""
## 3. Create a fiddler callback handler instance
"""
logger.info("## 3. Create a fiddler callback handler instance")


fiddler_handler = FiddlerCallbackHandler(
    url=URL,
    org=ORG_NAME,
    project=PROJECT_NAME,
    model=MODEL_NAME,
    api_key=AUTH_TOKEN,
)

"""
## Example 1 : Basic Chain
"""
logger.info("## Example 1 : Basic Chain")


llm = ChatOllama(temperature=0, streaming=True, callbacks=[fiddler_handler])
output_parser = StrOutputParser()

chain = llm | output_parser

chain.invoke("How far is moon from earth?")

chain.invoke("What is the temperature on Mars?")
chain.invoke("How much is 2 + 200000?")
chain.invoke("Which movie won the oscars this year?")
chain.invoke("Can you write me a poem about insomnia?")
chain.invoke("How are you doing today?")
chain.invoke("What is the meaning of life?")

"""
## Example 2 : Chain with prompt templates
"""
logger.info("## Example 2 : Chain with prompt templates")


examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

llm = ChatOllama(temperature=0, streaming=True, callbacks=[fiddler_handler])

chain = final_prompt | llm

chain.invoke({"input": "What's the square of a triangle?"})

logger.info("\n\n[DONE]", bright=True)

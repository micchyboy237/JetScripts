from jet.logger import logger
from langchain_ai21 import AI21ContextualAnswers
from langchain_ai21 import AI21LLM
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
---
sidebar_label: AI21 Labs
---

# AI21LLM

:::caution This service is deprecated.
See [this page](https://python.langchain.com/docs/integrations/chat/ai21/) for the updated ChatAI21 object. :::

This example goes over how to use LangChain to interact with `AI21` Jurassic models. To use the Jamba model, use the [ChatAI21 object](https://python.langchain.com/docs/integrations/chat/ai21/) instead.

[See a full list of AI21 models and tools on LangChain.](https://pypi.org/project/langchain-ai21/)

## Installation
"""
logger.info("# AI21LLM")

# !pip install -qU langchain-ai21

"""
## Environment Setup

We'll need to get a [AI21 API key](https://docs.ai21.com/) and set the `AI21_API_KEY` environment variable:
"""
logger.info("## Environment Setup")

# from getpass import getpass

if "AI21_API_KEY" not in os.environ:
#     os.environ["AI21_API_KEY"] = getpass()

"""
## Usage
"""
logger.info("## Usage")


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

model = AI21LLM(model="j2-ultra")

chain = prompt | model

chain.invoke({"question": "What is LangChain?"})

"""
# AI21 Contextual Answer

You can use AI21's contextual answers model to receives text or document, serving as a context,
and a question and returns an answer based entirely on this context.

This means that if the answer to your question is not in the document,
the model will indicate it (instead of providing a false answer)
"""
logger.info("# AI21 Contextual Answer")


tsm = AI21ContextualAnswers()

response = tsm.invoke(input={"context": "Your context", "question": "Your question"})

"""
You can also use it with chains and output parsers and vector DBs
"""
logger.info("You can also use it with chains and output parsers and vector DBs")


tsm = AI21ContextualAnswers()
chain = tsm | StrOutputParser()

response = chain.invoke(
    {"context": "Your context", "question": "Your question"},
)

logger.info("\n\n[DONE]", bright=True)
from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_gradient import ChatGradient
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
sidebar_label: DigitalOcean Gradient
---

# ChatGradient

This will help you getting started with DigitalOcean Gradient Chat Models.

## Overview
### Integration details

| Class | Package | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |
| [DigitalOcean Gradient](https://python.langchain.com/docs/api_reference/llms/langchain_gradient.llms.LangchainGradient/) | [langchain-gradient](https://python.langchain.com/docs/api_reference/langchain-gradient_api_reference/) | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-gradient?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-gradient?style=flat-square&label=%20) |


## Setup

langchain-gradient uses DigitalOcean Gradient Platform.  

Create an account on DigitalOcean, acquire a `DIGITALOCEAN_INFERENCE_KEY` API key from the Gradient Platform, and install the `langchain-gradient` integration package.

### Credentials

Head to [DigitalOcean Login](https://cloud.digitalocean.com/login) 

1. Sign up/Login to DigitalOcean Cloud Console
2. Go to the Gradient Platform and navigate to Serverless Inference.
3. Click on Create model access key, enter a name, and create the key.

Once you've done this set the `DIGITALOCEAN_INFERENCE_KEY` environment variable:
"""
logger.info("# ChatGradient")

# import getpass

if not os.getenv("DIGITALOCEAN_INFERENCE_KEY"):
#     os.environ["DIGITALOCEAN_INFERENCE_KEY"] = getpass.getpass(
        "Enter your DIGITALOCEAN_INFERENCE_KEY API key: "
    )

"""
If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
### Installation

The DigitalOcean Gradient integration lives in the `langchain-gradient` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-gradient

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatGradient(
    model="llama3.3-70b-instruct",
)

"""
## Invocation
"""
logger.info("## Invocation")

messages = [
    (
        "system",
        "You are a creative storyteller. Continue any story prompt you receive in an engaging and imaginative way.",
    ),
    (
        "human",
        "Once upon a time, in a village at the edge of a mysterious forest, a young girl named Mira found a glowing stone...",
    ),
]
ai_msg = llm.invoke(messages)
ai_msg

logger.debug(ai_msg.content)

"""
## Chaining

We can chain our model with a prompt template like so:
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate(
    [
        (
            "system",
            'You are a knowledgeable assistant. Carefully read the provided context and answer the user\'s question. If the answer is present in the context, cite the relevant sentence. If not, reply with "Not found in context."',
        ),
        ("human", "Context: {context}\nQuestion: {question}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "context": (
            "The Eiffel Tower is located in Paris and was completed in 1889. "
            "It was designed by Gustave Eiffel's engineering company. "
            "The tower is one of the most recognizable structures in the world. "
            "The Statue of Liberty was a gift from France to the United States."
        ),
        "question": "Who designed the Eiffel Tower and when was it completed?",
    }
)

"""
## API reference

For detailed documentation of all ChatGradient features and configurations head to the API reference.
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
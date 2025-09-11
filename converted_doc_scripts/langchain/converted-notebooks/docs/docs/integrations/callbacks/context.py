from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.callbacks.context_callback import ContextCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
ChatPromptTemplate,
HumanMessagePromptTemplate,
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
# Context

>[Context](https://context.ai/) provides user analytics for LLM-powered products and features.

With `Context`, you can start understanding your users and improving their experiences in less than 30 minutes.

In this guide we will show you how to integrate with Context.

## Installation and Setup
"""
logger.info("# Context")

# %pip install --upgrade --quiet  langchain langchain-ollama langchain-community context-python

"""
### Getting API Credentials

To get your Context API token:

1. Go to the settings page within your Context account (https://with.context.ai/settings).
2. Generate a new API Token.
3. Store this token somewhere secure.

### Setup Context

To use the `ContextCallbackHandler`, import the handler from Langchain and instantiate it with your Context API token.

Ensure you have installed the `context-python` package before using the handler.
"""
logger.info("### Getting API Credentials")



token = os.environ["CONTEXT_API_TOKEN"]

context_callback = ContextCallbackHandler(token)

"""
## Usage
### Context callback within a chat model

The Context callback handler can be used to directly record transcripts between users and AI assistants.
"""
logger.info("## Usage")



token = os.environ["CONTEXT_API_TOKEN"]

chat = ChatOllama(
    headers={"user_id": "123"}, temperature=0, callbacks=[ContextCallbackHandler(token)]
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(content="I love programming."),
]

logger.debug(chat(messages))

"""
### Context callback within Chains

The Context callback handler can also be used to record the inputs and outputs of chains. Note that intermediate steps of the chain are not recorded - only the starting inputs and final outputs.

__Note:__ Ensure that you pass the same context object to the chat model and the chain.

Wrong:
> ```python
> chat = ChatOllama(model="llama3.2")])
> chain = LLMChain(llm=chat, prompt=chat_prompt_template, callbacks=[ContextCallbackHandler(token)])
> ```

Correct:
>```python
>handler = ContextCallbackHandler(token)
>chat = ChatOllama(model="llama3.2")
>chain = LLMChain(llm=chat, prompt=chat_prompt_template, callbacks=[callback])
>```
"""
logger.info("### Context callback within Chains")



token = os.environ["CONTEXT_API_TOKEN"]

human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="What is a good name for a company that makes {product}?",
        input_variables=["product"],
    )
)
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
callback = ContextCallbackHandler(token)
chat = ChatOllama(model="llama3.2")
chain = LLMChain(llm=chat, prompt=chat_prompt_template, callbacks=[callback])
logger.debug(chain.run("colorful socks"))

logger.info("\n\n[DONE]", bright=True)
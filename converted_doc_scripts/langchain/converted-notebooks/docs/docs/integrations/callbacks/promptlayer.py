from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.callbacks.promptlayer_callback import (
    PromptLayerCallbackHandler,
)
from langchain_community.llms import GPT4All
from langchain_core.messages import HumanMessage
import os
import promptlayer  # Don't forget this 🍰
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
# PromptLayer

>[PromptLayer](https://docs.promptlayer.com/introduction) is a platform for prompt engineering. It also helps with the LLM observability to visualize requests, version prompts, and track usage.
>
>While `PromptLayer` does have LLMs that integrate directly with LangChain (e.g. [`PromptLayerOllama`](/docs/integrations/llms/promptlayer_ollama)), using a callback is the recommended way to integrate `PromptLayer` with LangChain.

In this guide, we will go over how to setup the `PromptLayerCallbackHandler`. 

See [PromptLayer docs](https://docs.promptlayer.com/languages/langchain) for more information.

## Installation and Setup
"""
logger.info("# PromptLayer")

# %pip install --upgrade --quiet  langchain-community promptlayer --upgrade

"""
### Getting API Credentials

If you do not have a PromptLayer account, create one on [promptlayer.com](https://www.promptlayer.com). Then get an API key by clicking on the settings cog in the navbar and
set it as an environment variable called `PROMPTLAYER_API_KEY`

## Usage

Getting started with `PromptLayerCallbackHandler` is fairly simple, it takes two optional arguments:
1. `pl_tags` - an optional list of strings that will be tracked as tags on PromptLayer.
2. `pl_id_callback` - an optional function that will take `promptlayer_request_id` as an argument. This ID can be used with all of PromptLayer's tracking features to track, metadata, scores, and prompt usage.

## Simple Ollama Example

In this simple example we use `PromptLayerCallbackHandler` with `ChatOllama`. We add a PromptLayer tag named `chatollama`
"""
logger.info("### Getting API Credentials")


chat_llm = ChatOllama(
    temperature=0,
    callbacks=[PromptLayerCallbackHandler(pl_tags=["chatollama"])],
)
llm_results = chat_llm.invoke(
    [
        HumanMessage(content="What comes after 1,2,3 ?"),
        HumanMessage(content="Tell me another joke?"),
    ]
)
logger.debug(llm_results)

"""
## GPT4All Example
"""
logger.info("## GPT4All Example")


model = GPT4All(model="./models/gpt4all-model.bin", n_ctx=512, n_threads=8)
callbacks = [PromptLayerCallbackHandler(pl_tags=["langchain", "gpt4all"])]

response = model.invoke(
    "Once upon a time, ",
    config={"callbacks": callbacks},
)

"""
## Full Featured Example

In this example, we unlock more of the power of `PromptLayer`.

PromptLayer allows you to visually create, version, and track prompt templates. Using the [Prompt Registry](https://docs.promptlayer.com/features/prompt-registry), we can programmatically fetch the prompt template called `example`.

We also define a `pl_id_callback` function which takes in the `promptlayer_request_id` and logs a score, metadata and links the prompt template used. Read more about tracking on [our docs](https://docs.promptlayer.com/features/prompt-history/request-id).
"""
logger.info("## Full Featured Example")


def pl_id_callback(promptlayer_request_id):
    logger.debug("prompt layer id ", promptlayer_request_id)
    promptlayer.track.score(
        request_id=promptlayer_request_id, score=100
    )  # score is an integer 0-100
    promptlayer.track.metadata(
        request_id=promptlayer_request_id, metadata={"foo": "bar"}
    )  # metadata is a dictionary of key value pairs that is tracked on PromptLayer
    promptlayer.track.prompt(
        request_id=promptlayer_request_id,
        prompt_name="example",
        prompt_input_variables={"product": "toasters"},
        version=1,
    )  # link the request to a prompt template


ollama_llm = ChatOllama(
    model_name="gpt-3.5-turbo-instruct",
    callbacks=[PromptLayerCallbackHandler(pl_id_callback=pl_id_callback)],
)

example_prompt = promptlayer.prompts.get("example", version=1, langchain=True)
ollama_llm.invoke(example_prompt.format(product="toasters"))

"""
That is all it takes! After setup all your requests will show up on the PromptLayer dashboard.
This callback also works with any LLM implemented on LangChain.
"""
logger.info(
    "That is all it takes! After setup all your requests will show up on the PromptLayer dashboard.")

logger.info("\n\n[DONE]", bright=True)

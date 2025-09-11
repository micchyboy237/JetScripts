from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain_community.callbacks.trubrics_callback import TrubricsCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
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
# Trubrics


>[Trubrics](https://trubrics.com) is an LLM user analytics platform that lets you collect, analyse and manage user
prompts & feedback on AI models.
>
>Check out [Trubrics repo](https://github.com/trubrics/trubrics-sdk) for more information on `Trubrics`.

In this guide, we will go over how to set up the `TrubricsCallbackHandler`.

## Installation and Setup
"""
logger.info("# Trubrics")

# %pip install --upgrade --quiet  trubrics langchain langchain-community

"""
### Getting Trubrics Credentials

If you do not have a Trubrics account, create one on [here](https://trubrics.streamlit.app/). In this tutorial, we will use the `default` project that is built upon account creation.

Now set your credentials as environment variables:
"""
logger.info("### Getting Trubrics Credentials")


os.environ["TRUBRICS_EMAIL"] = "***@***"
os.environ["TRUBRICS_PASSWORD"] = "***"


"""
### Usage

The `TrubricsCallbackHandler` can receive various optional arguments. See [here](https://trubrics.github.io/trubrics-sdk/platform/user_prompts/#saving-prompts-to-trubrics) for kwargs that can be passed to Trubrics prompts.

```python
class TrubricsCallbackHandler(BaseCallbackHandler):

    """
logger.info("### Usage")
    Callback handler for Trubrics.
    
    Args:
        project: a trubrics project, default project is "default"
        email: a trubrics account email, can equally be set in env variables
        password: a trubrics account password, can equally be set in env variables
        **kwargs: all other kwargs are parsed and set to trubrics prompt variables, or added to the `metadata` dict
    """
```

## Examples

# Here are two examples of how to use the `TrubricsCallbackHandler` with Langchain [LLMs](/docs/how_to#llms) or [Chat Models](/docs/how_to#chat-models). We will use Ollama models, so set your `OPENAI_API_KEY` key here:
"""
logger.info("## Examples")

# os.environ["OPENAI_API_KEY"] = "sk-***"

"""
### 1. With an LLM
"""
logger.info("### 1. With an LLM")


llm = Ollama(callbacks=[TrubricsCallbackHandler()])

res = llm.generate(["Tell me a joke", "Write me a poem"])

logger.debug("--> GPT's joke: ", res.generations[0][0].text)
logger.debug()
logger.debug("--> GPT's poem: ", res.generations[1][0].text)

"""
### 2. With a chat model
"""
logger.info("### 2. With a chat model")


chat_llm = ChatOllama(
    callbacks=[
        TrubricsCallbackHandler(
            project="default",
            tags=["chat model"],
            user_id="user-id-1234",
            some_metadata={"hello": [1, 2]},
        )
    ]
)

chat_res = chat_llm.invoke(
    [
        SystemMessage(content="Every answer of yours must be about Ollama."),
        HumanMessage(content="Tell me a joke"),
    ]
)

logger.debug(chat_res.content)

logger.info("\n\n[DONE]", bright=True)
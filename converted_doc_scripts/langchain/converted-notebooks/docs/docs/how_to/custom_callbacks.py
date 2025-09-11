from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
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
# How to create custom callback handlers

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Callbacks](/docs/concepts/callbacks)

:::

LangChain has some built-in callback handlers, but you will often want to create your own handlers with custom logic.

To create a custom callback handler, we need to determine the [event(s)](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html#langchain-core-callbacks-base-basecallbackhandler) we want our callback handler to handle as well as what we want our callback handler to do when the event is triggered. Then all we need to do is attach the callback handler to the object, for example via [the constructor](/docs/how_to/callbacks_constructor) or [at runtime](/docs/how_to/callbacks_runtime).

In the example below, we'll implement streaming with a custom handler.

In our custom callback handler `MyCustomHandler`, we implement the `on_llm_new_token` handler to print the token we have just received. We then attach our custom handler to the model object as a constructor callback.
"""
logger.info("# How to create custom callback handlers")

# %pip install -qU langchain jet.adapters.langchain.chat_ollama

# import getpass

# os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()



class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        logger.debug(f"My custom handler, token: {token}")


prompt = ChatPromptTemplate.from_messages(["Tell me a joke about {animal}"])

model = ChatOllama(
    model="claude-3-7-sonnet-20250219", streaming=True, callbacks=[MyCustomHandler()]
)

chain = prompt | model

response = chain.invoke({"animal": "bears"})

"""
You can see [this reference page](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html#langchain-core-callbacks-base-basecallbackhandler) for a list of events you can handle. Note that the `handle_chain_*` events run for most LCEL runnables.

## Next steps

You've now learned how to create your own custom callback handlers.

Next, check out the other how-to guides in this section, such as [how to attach callbacks to a runnable](/docs/how_to/callbacks_attach).
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)
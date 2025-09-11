from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Dict, List
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
# How to propagate callbacks  constructor

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Callbacks](/docs/concepts/callbacks)
- [Custom callback handlers](/docs/how_to/custom_callbacks)

:::

Most LangChain modules allow you to pass `callbacks` directly into the constructor (i.e., initializer). In this case, the callbacks will only be called for that instance (and any nested runs).

:::warning
Constructor callbacks are scoped only to the object they are defined on. They are **not** inherited by children of the object. This can lead to confusing behavior,
and it's generally better to pass callbacks as a run time argument.
:::

Here's an example:
"""
logger.info("# How to propagate callbacks  constructor")

# %pip install -qU langchain jet.adapters.langchain.chat_ollama

# import getpass

# os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()




class LoggingHandler(BaseCallbackHandler):
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        logger.debug("Chat model started")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        logger.debug(f"Chat model ended, response: {response}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        logger.debug(f"Chain {serialized.get('name')} started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        logger.debug(f"Chain ended, outputs: {outputs}")


callbacks = [LoggingHandler()]
llm = ChatOllama(model="llama3.2")
prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")

chain = prompt | llm

chain.invoke({"number": "2"})

"""
You can see that we only see events from the chat model run - no chain events from the prompt or broader chain.

## Next steps

You've now learned how to pass callbacks into a constructor.

Next, check out the other how-to guides in this section, such as how to [pass callbacks at runtime](/docs/how_to/callbacks_runtime).
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)
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
# How to pass callbacks in at runtime

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Callbacks](/docs/concepts/callbacks)
- [Custom callback handlers](/docs/how_to/custom_callbacks)

:::

In many cases, it is advantageous to pass in handlers instead when running the object. When we pass through [`CallbackHandlers`](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html#langchain-core-callbacks-base-basecallbackhandler) using the `callbacks` keyword arg when executing a run, those callbacks will be issued by all nested objects involved in the execution. For example, when a handler is passed through to an Agent, it will be used for all callbacks related to the agent and all the objects involved in the agent's execution, in this case, the Tools and LLM.

This prevents us from having to manually attach the handlers to each individual nested object. Here's an example:
"""
logger.info("# How to pass callbacks in at runtime")

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

chain.invoke({"number": "2"}, config={"callbacks": callbacks})

"""
If there are already existing callbacks associated with a module, these will run in addition to any passed in at runtime.

## Next steps

You've now learned how to pass callbacks at runtime.

Next, check out the other how-to guides in this section, such as how to [pass callbacks into a module constructor](/docs/how_to/custom_callbacks).
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)
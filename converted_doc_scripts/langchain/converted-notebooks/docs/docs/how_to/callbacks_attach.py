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
# How to attach callbacks to a runnable

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Callbacks](/docs/concepts/callbacks)
- [Custom callback handlers](/docs/how_to/custom_callbacks)
- [Chaining runnables](/docs/how_to/sequence)
- [Attach runtime arguments to a Runnable](/docs/how_to/binding)

:::

If you are composing a chain of runnables and want to reuse callbacks across multiple executions, you can attach callbacks with the [`.with_config()`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.with_config) method. This saves you the need to pass callbacks in each time you invoke the chain.

:::important

`with_config()` binds a configuration which will be interpreted as **runtime** configuration. So these callbacks will propagate to all child components.
:::

Here's an example:
"""
logger.info("# How to attach callbacks to a runnable")

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

chain_with_callbacks = chain.with_config(callbacks=callbacks)

chain_with_callbacks.invoke({"number": "2"})

"""
The bound callbacks will run for all nested module runs.

## Next steps

You've now learned how to attach callbacks to a chain.

Next, check out the other how-to guides in this section, such as how to [pass callbacks in at runtime](/docs/how_to/callbacks_runtime).
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)
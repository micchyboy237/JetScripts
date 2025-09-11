from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.prompts.chat import (
BaseChatPromptTemplate,
BaseMessage,
BaseMessagePromptTemplate,
)
from typing import Union
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
# MESSAGE_COERCION_FAILURE

Instead of always requiring instances of `BaseMessage`, several modules in LangChain take `MessageLikeRepresentation`, which is defined as:
"""
logger.info("# MESSAGE_COERCION_FAILURE")



MessageLikeRepresentation = Union[
    Union[BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate],
    tuple[
        Union[str, type],
        Union[str, list[dict], list[object]],
    ],
    str,
]

"""
These include Ollama style message objects (`{ role: "user", content: "Hello world!" }`),
tuples, and plain strings (which are converted to [`HumanMessages`](/docs/concepts/messages/#humanmessage)).

If one of these modules receives a value outside of one of these formats, you will receive an error like the following:
"""
logger.info("These include Ollama style message objects (`{ role: "user", content: "Hello world!" }`),")


uncoercible_message = {"role": "HumanMessage", "random_field": "random value"}

model = ChatOllama(model="llama3.2")

model.invoke([uncoercible_message])

"""
## Troubleshooting

The following may help resolve this error:

- Ensure that all inputs to chat models are an array of LangChain message classes or a supported message-like.
  - Check that there is no stringification or other unexpected transformation occurring.
- Check the error's stack trace and add log or debugger statements.


"""
logger.info("## Troubleshooting")

logger.info("\n\n[DONE]", bright=True)
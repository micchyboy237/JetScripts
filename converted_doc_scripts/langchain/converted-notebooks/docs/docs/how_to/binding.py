from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
sidebar_position: 2
keywords: [RunnableBinding, LCEL]
---

# How to add default invocation args to a Runnable

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [LangChain Expression Language (LCEL)](/docs/concepts/lcel)
- [Chaining runnables](/docs/how_to/sequence/)
- [Tool calling](/docs/how_to/tool_calling)

:::

Sometimes we want to invoke a [`Runnable`](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html) within a [RunnableSequence](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableSequence.html) with constant arguments that are not part of the output of the preceding Runnable in the sequence, and which are not part of the user input. We can use the [`Runnable.bind()`](https://python.langchain.com/api_reference/langchain_core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.bind) method to set these arguments ahead of time.

## Binding stop sequences

Suppose we have a simple prompt + model chain:
"""
logger.info("# How to add default invocation args to a Runnable")

# %pip install -qU langchain jet.adapters.langchain.chat_ollama

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write out the following equation using algebraic symbols then solve it. Use the format\n\nEQUATION:...\nSOLUTION:...\n\n",
        ),
        ("human", "{equation_statement}"),
    ]
)

model = ChatOllama(model="llama3.2")

runnable = (
    {"equation_statement": RunnablePassthrough()} | prompt | model | StrOutputParser()
)

logger.debug(runnable.invoke("x raised to the third plus seven equals 12"))

"""
and want to call the model with certain `stop` words so that we shorten the output as is useful in certain types of prompting techniques. While we can pass some arguments into the constructor, other runtime args use the `.bind()` method as follows:
"""
logger.info("and want to call the model with certain `stop` words so that we shorten the output as is useful in certain types of prompting techniques. While we can pass some arguments into the constructor, other runtime args use the `.bind()` method as follows:")

runnable = (
    {"equation_statement": RunnablePassthrough()}
    | prompt
    | model.bind(stop="SOLUTION")
    | StrOutputParser()
)

logger.debug(runnable.invoke("x raised to the third plus seven equals 12"))

"""
What you can bind to a Runnable will depend on the extra parameters you can pass when invoking it.

## Attaching Ollama tools

Another common use-case is tool calling. While you should generally use the [`.bind_tools()`](/docs/how_to/tool_calling) method for tool-calling models, you can also bind provider-specific args directly if you want lower level control:
"""
logger.info("## Attaching Ollama tools")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

model = ChatOllama(model="llama3.2").bind(tools=tools)
model.invoke("What's the weather in SF, NYC and LA?")

"""
## Next steps

You now know how to bind runtime arguments to a Runnable.

To learn more, see the other how-to guides on runnables in this section, including:

- [Using configurable fields and alternatives](/docs/how_to/configure) to change parameters of a step in a chain, or even swap out entire steps, at runtime
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)
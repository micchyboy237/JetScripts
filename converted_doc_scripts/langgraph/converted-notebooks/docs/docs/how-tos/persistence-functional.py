from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
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
# How to add thread-level persistence (functional API)

!!! info "Prerequisites"

    This guide assumes familiarity with the following:
    
    - [Functional API](../../concepts/functional_api/)
    - [Persistence](../../concepts/persistence/)
    - [Memory](../../concepts/memory/)
    - [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

!!! info "Not needed for LangGraph API users"

    If you're using the LangGraph API, you needn't manually implement a checkpointer. The API automatically handles checkpointing for you. This guide is relevant when implementing LangGraph in your own custom server.

Many AI applications need memory to share context across multiple interactions on the same [thread](../../concepts/persistence#threads) (e.g., multiple turns of a conversation). In LangGraph functional API, this kind of memory can be added to any [entrypoint()][langgraph.func.entrypoint] workflow using [thread-level persistence](https://langchain-ai.github.io/langgraph/concepts/persistence).

When creating a LangGraph workflow, you can set it up to persist its results by using a [checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#basecheckpointsaver):


1. Create an instance of a checkpointer:

    ```python
    
    checkpointer = InMemorySaver()       
    ```

2. Pass `checkpointer` instance to the `entrypoint()` decorator:

    ```python
    
    @entrypoint(checkpointer=checkpointer)
    def workflow(inputs)
        ...
    ```

3. Optionally expose `previous` parameter in the workflow function signature:

    ```python
    @entrypoint(checkpointer=checkpointer)
    def workflow(
        inputs,
        *,
        # you can optionally specify `previous` in the workflow function signature
        # to access the return value from the workflow as of the last execution
        previous
    ):
        previous = previous or []
        combined_inputs = previous + inputs
        result = do_something(combined_inputs)
        ...
    ```

4. Optionally choose which values will be returned from the workflow and which will be saved by the checkpointer as `previous`:

    ```python
    @entrypoint(checkpointer=checkpointer)
    def workflow(inputs, *, previous):
        ...
        result = do_something(...)
        return entrypoint.final(value=result, save=combine(inputs, result))
    ```

This guide shows how you can add thread-level persistence to your workflow.

!!! tip "Note"

    If you need memory that is __shared__ across multiple conversations or users (cross-thread persistence), check out this [how-to guide](../cross-thread-persistence-functional).

!!! tip "Note"

    If you need to add thread-level persistence to a `StateGraph`, check out this [how-to guide](../persistence).

## Setup

First we need to install the packages required
"""
logger.info("# How to add thread-level persistence (functional API)")

# %%capture --no-stderr
# %pip install --quiet -U langgraph jet.adapters.langchain.chat_ollama

"""
Next, we need to set API key for Ollama(the LLM we will use).
"""
logger.info("Next, we need to set API key for Ollama(the LLM we will use).")

# import getpass


# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("ANTHROPIC_API_KEY")

"""
<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Example: simple chatbot with short-term memory

We will be using a workflow with a single task that calls a [chat model](https://python.langchain.com/docs/concepts/chat_models/).

Let's first define the model we'll be using:
"""
logger.info("## Example: simple chatbot with short-term memory")


model = ChatOllama(model="llama3.2")

"""
Now we can define our task and workflow. To add in persistence, we need to pass in a [Checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver) to the [entrypoint()][langgraph.func.entrypoint] decorator.
"""
logger.info(
    "Now we can define our task and workflow. To add in persistence, we need to pass in a [Checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver) to the [entrypoint()][langgraph.func.entrypoint] decorator.")


@task
def call_model(messages: list[BaseMessage]):
    response = model.invoke(messages)
    return response


checkpointer = InMemorySaver()


@entrypoint(checkpointer=checkpointer)
def workflow(inputs: list[BaseMessage], *, previous: list[BaseMessage]):
    if previous:
        inputs = add_messages(previous, inputs)

    response = call_model(inputs).result()
    return entrypoint.final(value=response, save=add_messages(inputs, response))


"""
If we try to use this workflow, the context of the conversation will be persisted across interactions:

!!! note Note

    If you're using LangGraph Platform or LangGraph Studio, you __don't need__ to pass checkpointer to the entrypoint decorator, since it's done automatically.

We can now interact with the agent and see that it remembers previous messages!
"""
logger.info("If we try to use this workflow, the context of the conversation will be persisted across interactions:")

config = {"configurable": {"thread_id": "1"}}
input_message = {"role": "user", "content": "hi! I'm bob"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_logger.debug()

"""
You can always resume previous threads:
"""
logger.info("You can always resume previous threads:")

input_message = {"role": "user", "content": "what's my name?"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_logger.debug()

"""
If we want to start a new conversation, we can pass in a different `thread_id`. Poof! All the memories are gone!
"""
logger.info("If we want to start a new conversation, we can pass in a different `thread_id`. Poof! All the memories are gone!")

input_message = {"role": "user", "content": "what's my name?"}
for chunk in workflow.stream(
    [input_message],
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
):
    chunk.pretty_logger.debug()

"""
!!! tip "Streaming tokens"

    If you would like to stream LLM tokens from your chatbot, you can use `stream_mode="messages"`. Check out this [how-to guide](../streaming-tokens) to learn more.
"""
logger.info("If you would like to stream LLM tokens from your chatbot, you can use `stream_mode="messages"`. Check out this [how-to guide](../streaming-tokens) to learn more.")

logger.info("\n\n[DONE]", bright=True)

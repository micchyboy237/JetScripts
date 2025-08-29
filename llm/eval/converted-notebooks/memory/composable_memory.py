import nest_asyncio
from llama_index.core.agent import FunctionCallingAgent, FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from jet.llm.ollama.base import Ollama
from jet.llm.ollama.base import OllamaEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import (
    VectorMemory,
    SimpleComposableMemory,
    ChatMemoryBuffer,
)
import json
from jet.transformers.object import make_serializable
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings, large_embed_model
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/memory/composable_memory.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Simple Composable Memory

# In this notebook, we demonstrate how to inject multiple memory sources into an agent. Specifically, we use the `SimpleComposableMemory` which is comprised of a `primary_memory` as well as potentially several secondary memory sources (stored in `secondary_memory_sources`). The main difference is that `primary_memory` will be used as the main chat buffer for the agent, where as any retrieved messages from `secondary_memory_sources` will be injected to the system prompt message only.
#
# Multiple memory sources may be of use for example in situations where you have a longer-term memory such as `VectorMemory` that you want to use in addition to the default `ChatMemoryBuffer`. What you'll see in this notebook is that with a `SimpleComposableMemory` you'll be able to effectively "load" the desired messages from long-term memory into the main memory (i.e. the `ChatMemoryBuffer`).

# How `SimpleComposableMemory` Works?

# We begin with the basic usage of the `SimpleComposableMemory`. Here we construct a `VectorMemory` as well as a default `ChatMemoryBuffer`. The `VectorMemory` will be our secondary memory source, whereas the `ChatMemoryBuffer` will be the main or primary one. To instantiate a `SimpleComposableMemory` object, we need to supply a `primary_memory` and (optionally) a list of `secondary_memory_sources`.

# ![SimpleComposableMemoryIllustration](https://d3ddy8balm3goa.cloudfront.net/llamaindex/simple-composable-memory.excalidraw.svg)


vector_memory = VectorMemory.from_defaults(
    vector_store=None,  # leave as None to use default in-memory vector store
    embed_model=OllamaEmbedding(model_name=large_embed_model),
    retriever_kwargs={"similarity_top_k": 1},
)

msgs = [
    ChatMessage.from_str("You are a SOMEWHAT helpful assistant.", "system"),
    ChatMessage.from_str("Bob likes burgers.", "user"),
    ChatMessage.from_str("Indeed, Bob likes apples.", "assistant"),
    ChatMessage.from_str("Alice likes apples.", "user"),
]
vector_memory.set(msgs)

chat_memory_buffer = ChatMemoryBuffer.from_defaults()

composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=chat_memory_buffer,
    secondary_memory_sources=[vector_memory],
)

logger.newline()
logger.debug("Primary Memory:")
logger.success(composable_memory.primary_memory.get())

logger.newline()
logger.debug("Secondary Memory:")
logger.success(composable_memory.secondary_memory_sources)

# `put()` messages into memory

# Since `SimpleComposableMemory` is itself a subclass of `BaseMemory`, we add messages to it in the same way as we do for other memory modules. Note that for `SimpleComposableMemory`, invoking `.put()` effectively calls `.put()` on all memory sources. In other words, the message gets added to `primary` and `secondary` sources.

msgs = [
    ChatMessage.from_str("You are a REALLY helpful assistant.", "system"),
    ChatMessage.from_str("Jerry likes juice.", "user"),
]

for m in msgs:
    composable_memory.put(m)

# `get()` messages from memory

# When `.get()` is invoked, we similarly execute all of the `.get()` methods of `primary` memory as well as all of the `secondary` sources. This leaves us with sequence of lists of messages that we have to must "compose" into a sensible single set of messages (to pass downstream to our agents). Special care must be applied here in general to ensure that the final sequence of messages are both sensible and conform to the chat APIs of the LLM provider.
#
# For `SimpleComposableMemory`, we **inject the messages from the `secondary` sources in the system message of the `primary` memory**. The rest of the message history of the `primary` source is left intact, and this composition is what is ultimately returned.

msgs = composable_memory.get("What does Bob like?")
logger.info("Response 1")
logger.success(json.dumps(make_serializable(msgs), indent=2))

# Successive calls to `get()`

# Successive calls of `get()` will simply replace the loaded `secondary` memory messages in the system prompt.

msgs = composable_memory.get("What does Alice like?")
logger.info("Response 2")
logger.success(json.dumps(make_serializable(msgs), indent=2))

# What if `get()` retrieves `secondary` messages that already exist in `primary` memory?

# In the event that messages retrieved from `secondary` memory already exist in `primary` memory, then these rather redundant secondary messages will not get added to the system message. In the below example, the message "Jerry likes juice." was `put` into all memory sources, so the system message is not altered.

msgs = composable_memory.get("What does Jerry like?")
logger.info("Response 3")
logger.success(json.dumps(make_serializable(msgs), indent=2))

# How to `reset` memory

# Similar to the other methods `put()` and `get()`, calling `reset()` will execute `reset()` on both the `primary` and `secondary` memory sources. If you want to reset only the `primary` then you should call the `reset()` method only from it.

# `reset()` only primary memory

composable_memory.primary_memory.reset()

logger.newline()
logger.debug("Primary Memory Reset:")
logger.success(composable_memory.primary_memory.get())

logger.newline()
logger.debug("Secondary Memory:")
logger.success(composable_memory.secondary_memory_sources[0].get(
    "What does Alice like?"))

# `reset()` all memory sources

composable_memory.reset()

composable_memory.secondary_memory_sources[0].get("What does Alice like?")
logger.newline()
logger.debug("Primary Memory Reset:")
logger.success(composable_memory.primary_memory.get())

logger.newline()
logger.debug("Secondary Memory Reset:")
logger.success(composable_memory.secondary_memory_sources[0].get(
    "What does Alice like?"))

# Use `SimpleComposableMemory` With An Agent

# Here we will use a `SimpleComposableMemory` with an agent and demonstrate how a secondary, long-term memory source can be used to use messages from on agent conversation as part of another conversation with another agent session.


nest_asyncio.apply()

# Define our memory modules

vector_memory = VectorMemory.from_defaults(
    vector_store=None,  # leave as None to use default in-memory vector store
    embed_model=OllamaEmbedding(model_name=large_embed_model),
    retriever_kwargs={"similarity_top_k": 2},
)

chat_memory_buffer = ChatMemoryBuffer.from_defaults()

composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=chat_memory_buffer,
    secondary_memory_sources=[vector_memory],
)

# Define our Agent


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def mystery(a: int, b: int) -> int:
    """Mystery function on two numbers"""
    return a**2 - b**2


multiply_tool = FunctionTool.from_defaults(fn=multiply)
mystery_tool = FunctionTool.from_defaults(fn=mystery)

llm = Ollama(model="llama3.1")
agent = FunctionCallingAgent.from_tools(
    [multiply_tool, mystery_tool],
    llm=llm,
    memory=composable_memory,
    verbose=True,
)

# Execute some function calls

# When `.chat()` is invoked, the messages are put into the composable memory, which we understand from the previous section implies that all the messages are put in both `primary` and `secondary` sources.

response = agent.chat("What is the mystery function on 5 and 6?")
logger.info("Agent w/ memory response 1")
logger.success(json.dumps(make_serializable(response), indent=2))

response = agent.chat("What happens if you multiply 2 and 3?")
logger.info("Agent w/ memory response 2")
logger.success(json.dumps(make_serializable(response), indent=2))

# New Agent Sessions

# Now that we've added the messages to our `vector_memory`, we can see the effect of having this memory be used with a new agent session versus when it is used. Specifically, we ask the new agents to "recall" the outputs of the function calls, rather than re-computing.

# An Agent without our past memory

llm = Ollama(model="llama3.1")
agent_without_memory = FunctionCallingAgent.from_tools(
    [multiply_tool, mystery_tool], llm=llm, verbose=True
)

response = agent_without_memory.chat(
    "What was the output of the mystery function on 5 and 6 again? Don't recompute."
)
logger.info("Agent w/o memory response 1")
logger.success(json.dumps(make_serializable(response), indent=2))

# An Agent with our past memory

# We see that the agent without access to the our past memory cannot complete the task. With this next agent we will indeed pass in our previous long-term memory (i.e., `vector_memory`). Note that we even use a fresh `ChatMemoryBuffer` which means there is no `chat_history` with this agent. Nonetheless, it will be able to retrieve from our long-term memory to get the past dialogue it needs.

llm = Ollama(model="llama3.1")

composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=ChatMemoryBuffer.from_defaults(),
    secondary_memory_sources=[
        vector_memory.model_copy(
            deep=False
        )  # using a copy here for illustration purposes
    ],
)

agent_with_memory = FunctionCallingAgent.from_tools(
    [multiply_tool, mystery_tool],
    llm=llm,
    memory=composable_memory,
    verbose=True,
)

agent_with_memory.chat_history  # an empty chat history

response = agent_with_memory.chat(
    "What was the output of the mystery function on 5 and 6 again? Don't recompute."
)
logger.info("Agent w/ past memory response 1")
logger.success(json.dumps(make_serializable(response), indent=2))

response = agent_with_memory.chat(
    "What was the output of the multiply function on 2 and 3 again? Don't recompute."
)
logger.info("Agent w/ past memory response 2")
logger.success(json.dumps(make_serializable(response), indent=2))


logger.info("Agent w/ past memory chat history")
logger.success(json.dumps(make_serializable(
    agent_with_memory.chat_history), indent=2))

# What happens under the hood with `.chat(user_input)`

# Under the hood, `.chat(user_input)` call effectively will call the memory's `.get()` method with `user_input` as the argument. As we learned in the previous section, this will ultimately return a composition of the `primary` and all of the `secondary` memory sources. These composed messages are what is being passed to the LLM's chat API as the chat history.

composable_memory = SimpleComposableMemory.from_defaults(
    primary_memory=ChatMemoryBuffer.from_defaults(),
    secondary_memory_sources=[
        vector_memory.model_copy(
            deep=False
        )  # copy for illustrative purposes to explain what
    ],
)

llm = Ollama(model="llama3.1")
agent_worker = FunctionCallingAgentWorker.from_tools(
    [multiply_tool, mystery_tool], llm=llm, verbose=True
)
agent_with_memory = agent_worker.as_agent(memory=composable_memory)


response = agent_with_memory.memory.get(
    "What was the output of the mystery function on 5 and 6 again? Don't recompute."
)

logger.info("Agent w/ tools and memory response 1")
logger.success(json.dumps(make_serializable(response), indent=2))

logger.info("\n\n[DONE]", bright=True)

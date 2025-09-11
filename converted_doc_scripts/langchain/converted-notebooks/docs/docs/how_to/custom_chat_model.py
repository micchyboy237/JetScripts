from jet.logger import logger
from langchain_core.callbacks import (
CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
AIMessage,
AIMessageChunk,
BaseMessage,
)
from langchain_core.messages import (
AIMessage,
BaseMessage,
FunctionMessage,
HumanMessage,
SystemMessage,
ToolMessage,
)
from langchain_core.messages import (
AIMessageChunk,
FunctionMessageChunk,
HumanMessageChunk,
SystemMessageChunk,
ToolMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
from typing import Any, Dict, Iterator, List, Optional
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
# How to create a custom chat model class

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models)

:::

In this guide, we'll learn how to create a custom [chat model](/docs/concepts/chat_models/) using LangChain abstractions.

Wrapping your LLM with the standard [`BaseChatModel`](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html) interface allow you to use your LLM in existing LangChain programs with minimal code modifications!

As a bonus, your LLM will automatically become a LangChain [Runnable](/docs/concepts/runnables/) and will benefit from some optimizations out of the box (e.g., batch via a threadpool), async support, the `astream_events` API, etc.

## Inputs and outputs

First, we need to talk about **[messages](/docs/concepts/messages/)**, which are the inputs and outputs of chat models.

### Messages

Chat models take messages as inputs and return a message as output. 

LangChain has a few [built-in message types](/docs/concepts/messages):

| Message Type          | Description                                                                                     |
|-----------------------|-------------------------------------------------------------------------------------------------|
| `SystemMessage`       | Used for priming AI behavior, usually passed in as the first of a sequence of input messages.   |
| `HumanMessage`        | Represents a message from a person interacting with the chat model.                             |
| `AIMessage`           | Represents a message from the chat model. This can be either text or a request to invoke a tool.|
| `FunctionMessage` / `ToolMessage` | Message for passing the results of tool invocation back to the model.               |
| `AIMessageChunk` / `HumanMessageChunk` / ... | Chunk variant of each type of message. |


:::note
`ToolMessage` and `FunctionMessage` closely follow Ollama's `function` and `tool` roles.

This is a rapidly developing field and as more models add function calling capabilities. Expect that there will be additions to this schema.
:::
"""
logger.info("# How to create a custom chat model class")


"""
### Streaming Variant

All the chat messages have a streaming variant that contains `Chunk` in the name.
"""
logger.info("### Streaming Variant")


"""
These chunks are used when streaming output from chat models, and they all define an additive property!
"""
logger.info("These chunks are used when streaming output from chat models, and they all define an additive property!")

AIMessageChunk(content="Hello") + AIMessageChunk(content=" World!")

"""
## Base Chat Model

Let's implement a chat model that echoes back the first `n` characters of the last message in the prompt!

To do so, we will inherit from `BaseChatModel` and we'll need to implement the following:

| Method/Property                    | Description                                                       | Required/Optional  |
|------------------------------------|-------------------------------------------------------------------|--------------------|
| `_generate`                        | Use to generate a chat result from a prompt                       | Required           |
| `_llm_type` (property)             | Used to uniquely identify the type of the model. Used for logging.| Required           |
| `_identifying_params` (property)   | Represent model parameterization for tracing purposes.            | Optional           |
| `_stream`                          | Use to implement streaming.                                       | Optional           |
| `_agenerate`                       | Use to implement a native async method.                           | Optional           |
| `_astream`                         | Use to implement async version of `_stream`.                      | Optional           |


:::tip
The `_astream` implementation uses `run_in_executor` to launch the sync `_stream` in a separate thread if `_stream` is implemented, otherwise it fallsback to use `_agenerate`.

You can use this trick if you want to reuse the `_stream` implementation, but if you're able to implement code that's natively async that's a better solution since that code will run with less overhead.
:::

### Implementation
"""
logger.info("## Base Chat Model")




class ChatParrotLink(BaseChatModel):
    """A custom chat model that echoes the first `parrot_buffer_length` characters
    of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = ChatParrotLink(parrot_buffer_length=2, model="bird-brain-001")
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    model_name: str = Field(alias="model")
    """The name of the model"""
    parrot_buffer_length: int
    """The number of characters from the last message of the prompt to be echoed."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        last_message = messages[-1]
        tokens = last_message.content[: self.parrot_buffer_length]
        ct_input_tokens = sum(len(message.content) for message in messages)
        ct_output_tokens = len(tokens)
        message = AIMessage(
            content=tokens,
            additional_kwargs={},  # Used to add additional payload to the message
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
                "model_name": self.model_name,
            },
            usage_metadata={
                "input_tokens": ct_input_tokens,
                "output_tokens": ct_output_tokens,
                "total_tokens": ct_input_tokens + ct_output_tokens,
            },
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        last_message = messages[-1]
        tokens = str(last_message.content[: self.parrot_buffer_length])
        ct_input_tokens = sum(len(message.content) for message in messages)

        for token in tokens:
            usage_metadata = UsageMetadata(
                {
                    "input_tokens": ct_input_tokens,
                    "output_tokens": 1,
                    "total_tokens": ct_input_tokens + 1,
                }
            )
            ct_input_tokens = 0
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=token, usage_metadata=usage_metadata)
            )

            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                response_metadata={"time_in_sec": 3, "model_name": self.model_name},
            )
        )
        if run_manager:
            run_manager.on_llm_new_token(token, chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model_name": self.model_name,
        }

"""
### Let's test it 🧪

The chat model will implement the standard `Runnable` interface of LangChain which many of the LangChain abstractions support!
"""
logger.info("### Let's test it 🧪")

model = ChatParrotLink(parrot_buffer_length=3, model="my_custom_model")

model.invoke(
    [
        HumanMessage(content="hello!"),
        AIMessage(content="Hi there human!"),
        HumanMessage(content="Meow!"),
    ]
)

model.invoke("hello")

model.batch(["hello", "goodbye"])

for chunk in model.stream("cat"):
    logger.debug(chunk.content, end="|")

"""
Please see the implementation of `_astream` in the model! If you do not implement it, then no output will stream.!
"""
logger.info("Please see the implementation of `_astream` in the model! If you do not implement it, then no output will stream.!")

for chunk in model.stream("cat"):
    logger.debug(chunk.content, end="|")

"""
Let's try to use the astream events API which will also help double check that all the callbacks were implemented!
"""
logger.info("Let's try to use the astream events API which will also help double check that all the callbacks were implemented!")

for event in model.stream_events("cat", version="v1"):
    logger.debug(event)

"""
## Contributing

We appreciate all chat model integration contributions. 

Here's a checklist to help make sure your contribution gets added to LangChain:

Documentation:

* The model contains doc-strings for all initialization arguments, as these will be surfaced in the [API Reference](https://python.langchain.com/api_reference/langchain/index.html).
* The class doc-string for the model contains a link to the model API if the model is powered by a service.

Tests:

* [ ] Add unit or integration tests to the overridden methods. Verify that `invoke`, `ainvoke`, `batch`, `stream` work if you've over-ridden the corresponding code.


Streaming (if you're implementing it):

* [ ] Implement the _stream method to get streaming working

Stop Token Behavior:

* [ ] Stop token should be respected
* [ ] Stop token should be INCLUDED as part of the response

Secret API Keys:

* [ ] If your model connects to an API it will likely accept API keys as part of its initialization. Use Pydantic's `SecretStr` type for secrets, so they don't get accidentally printed out when folks print the model.


Identifying Params:

* [ ] Include a `model_name` in identifying params


Optimizations:

Consider providing native async support to reduce the overhead from the model!
 
* [ ] Provided a native async of `_agenerate` (used by `ainvoke`)
* [ ] Provided a native async of `_astream` (used by `astream`)

## Next steps

You've now learned how to create your own custom chat models.

Next, check out the other how-to guides chat models in this section, like [how to get a model to return structured output](/docs/how_to/structured_output) or [how to track chat model token usage](/docs/how_to/chat_token_usage_tracking).
"""
logger.info("## Contributing")

logger.info("\n\n[DONE]", bright=True)
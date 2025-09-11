from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_writer import ChatWriter
from pydantic import BaseModel, Field
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
# Chat Writer

This notebook provides a quick overview for getting started with Writer [chat](/docs/concepts/chat_models/).

Writer has several chat models. You can find information about their latest models and their costs, context windows, and supported input types in the [Writer docs](https://dev.writer.com/home).


## Overview

### Integration details
| Class                                                                                                                    | Package          | Local | Serializable | JS support |                                        Package downloads                                         |                                        Package latest                                         |
|:-------------------------------------------------------------------------------------------------------------------------|:-----------------| :---: | :---: |:----------:|:------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|
| [ChatWriter](https://github.com/writer/langchain-writer/blob/main/langchain_writer/chat_models.py#L308) | [langchain-writer](https://pypi.org/project/langchain-writer/) |      ❌       |                                       ❌                                       | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-writer?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-writer?style=flat-square&label=%20) |
### Model features
| [Tool calling](/docs/how_to/tool_calling) | Structured output | JSON mode | Image input | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async |         [Token usage](/docs/how_to/chat_token_usage_tracking/)          | Logprobs |
| :---: |:-----------------:| :---: | :---: |  :---: | :---: | :---: | :---: |:--------------------------------:|:--------:|
| ✅ |         ❌         | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |                ✅                 |    ❌     |

### Credentials

Sign up for [Writer AI Studio](https://app.writer.com/aistudio/signup?utm_campaign=devrel) and follow this [Quickstart](https://dev.writer.com/api-guides/quickstart) to obtain an API key. Then, set the WRITER_API_KEY environment variable:
"""
logger.info("# Chat Writer")

# import getpass

if not os.getenv("WRITER_API_KEY"):
#     os.environ["WRITER_API_KEY"] = getpass.getpass("Enter your Writer API key: ")

"""
If you want to get automated tracing of your model calls, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing of your model calls, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
### Installation

`ChatWriter` is available from the `langchain-writer` package. Install it with:
"""
logger.info("### Installation")

# %pip install -qU langchain-writer

"""
### Instantiation

Now we can instantiate our model object in order to generate chat completions:
"""
logger.info("### Instantiation")


llm = ChatWriter(
    model="palmyra-x-004",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

"""
## Usage

To use the model, you pass in a list of messages and call the `invoke` method:
"""
logger.info("## Usage")

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg

"""
Then, you can access the content of the message:
"""
logger.info("Then, you can access the content of the message:")

logger.debug(ai_msg.content)

"""
## Streaming

You can also stream the response. First, create a stream:
"""
logger.info("## Streaming")

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming. Sing a song about it"),
]
ai_stream = llm.stream(messages)
ai_stream

"""
Then, iterate over the stream to get the chunks:
"""
logger.info("Then, iterate over the stream to get the chunks:")

for chunk in ai_stream:
    logger.debug(chunk.content, end="")

"""
## Tool calling

Writer models like Palmyra X 004 support [tool calling](https://dev.writer.com/api-guides/tool-calling), which lets you describe tools and their arguments. The model will return a JSON object with a tool to invoke and the inputs to that tool.

### Binding tools

With `ChatWriter.bind_tools`, you can easily pass in Pydantic classes, dictionary schemas, LangChain tools, or even functions as tools to the model. Under the hood, these are converted to tool schemas, which look like this:
```
{
    "name": "...",
    "description": "...",
    "parameters": {...}  # JSONSchema
}
```
These are passed in every model invocation.

For example, to use a tool that gets the weather in a given location, you can define a Pydantic class and pass it to `ChatWriter.bind_tools`:
"""
logger.info("## Tool calling")



class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm.bind_tools([GetWeather])

"""
Then, you can invoke the model with the tool:
"""
logger.info("Then, you can invoke the model with the tool:")

ai_msg = llm.invoke(
    "what is the weather like in New York City",
)
ai_msg

"""
Finally, you can access the tool calls and proceed to execute your functions:
"""
logger.info("Finally, you can access the tool calls and proceed to execute your functions:")

logger.debug(ai_msg.tool_calls)

"""
### A note on tool binding

The `ChatWriter.bind_tools()` method does not create a new instance with bound tools, but stores the received `tools` and `tool_choice` in the initial class instance attributes to pass them as parameters during the Palmyra LLM call while using `ChatWriter` invocation. This approach allows the support of different tool types, e.g. `function` and `graph`. `Graph` is one of the remotely called Writer Palmyra tools. For further information, visit our [docs](https://dev.writer.com/api-guides/knowledge-graph#knowledge-graph). 

For more information about tool usage in LangChain, visit the [LangChain tool calling documentation](https://python.langchain.com/docs/concepts/tool_calling/).

## Batching

You can also batch requests and set the `max_concurrency`:
"""
logger.info("### A note on tool binding")

ai_batch = llm.batch(
    [
        "How to cook pancakes?",
        "How to compose poem?",
        "How to run faster?",
    ],
    config={"max_concurrency": 3},
)
ai_batch

"""
Then, iterate over the batch to get the results:
"""
logger.info("Then, iterate over the batch to get the results:")

for batch in ai_batch:
    logger.debug(batch.content)
    logger.debug("-" * 100)

"""
## Asynchronous usage

All features above (invocation, streaming, batching, tools calling) also support asynchronous usage.

## Prompt templates

[Prompt templates](https://python.langchain.com/docs/concepts/prompt_templates/) help to translate user input and parameters into instructions for a language model. You can use `ChatWriter` with a prompt template like so:
"""
logger.info("## Asynchronous usage")


prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

"""
## API reference
For detailed documentation of all ChatWriter features and configurations, head to the [API reference](https://python.langchain.com/api_reference/writer/chat_models/langchain_writer.chat_models.ChatWriter.html#langchain_writer.chat_models.ChatWriter).

## Additional resources
You can find information about Writer's models (including costs, context windows, and supported input types) and tools in the [Writer docs](https://dev.writer.com/home).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
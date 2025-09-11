from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_naver import ChatClovaX
from pydantic import BaseModel, Field
from typing import Optional
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
sidebar_label: Naver
---

# ChatClovaX

This notebook provides a quick overview for getting started with Naver’s HyperCLOVA X [chat models](https://python.langchain.com/docs/concepts/chat_models) via CLOVA Studio. For detailed documentation of all ChatClovaX features and configurations head to the [API reference](https://guide.ncloud-docs.com/docs/clovastudio-dev-langchain).

[CLOVA Studio](http://clovastudio.ncloud.com/) has several chat models. You can find information about the latest models, including their costs, context windows, and supported input types, in the CLOVA Studio Guide [documentation](https://guide.ncloud-docs.com/docs/clovastudio-model).

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- |:-----:| :---: |:------------------------------------------------------------------------:| :---: | :---: |
| [ChatClovaX](https://guide.ncloud-docs.com/docs/clovastudio-dev-langchain#HyperCLOVAX%EB%AA%A8%EB%8D%B8%EC%9D%B4%EC%9A%A9) | [langchain-naver](https://pypi.org/project/langchain-naver/) |   ❌   | ❌ |                                    ❌                                     | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_naver?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_naver?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling/) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
|:------------------------------------------:| :---: | :---: | :---: |  :---: | :---: |:-----------------------------------------------------:| :---: |:------------------------------------------------------:|:----------------------------------:|
|✅| ✅ | ❌ | ✅ | ❌ | ❌ |                          ✅                            | ✅ |                           ✅                            |                 ❌                  |

## Setup

Before using the chat model, you must go through the four steps below.

1. Creating [NAVER Cloud Platform](https://www.ncloud.com/) account
2. Apply to use [CLOVA Studio](https://www.ncloud.com/product/aiService/clovaStudio)
3. Create a CLOVA Studio Test App or Service App of a model to use (See [here](https://guide.ncloud-docs.com/docs/clovastudio-playground-testapp).)
4. Issue a Test or Service API key (See [here](https://api.ncloud-docs.com/docs/ai-naver-clovastudio-summary#API%ED%82%A4).)

### Credentials

Set the `CLOVASTUDIO_API_KEY` environment variable with your API key.

You can add them to your environment variables as below:

``` bash
export CLOVASTUDIO_API_KEY="your-api-key-here"
```
"""
logger.info("# ChatClovaX")

# import getpass

if not os.getenv("CLOVASTUDIO_API_KEY"):
#     os.environ["CLOVASTUDIO_API_KEY"] = getpass.getpass(
        "Enter your CLOVA Studio API Key: "
    )

"""
To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:")



"""
### Installation

The LangChain Naver integration lives in the `langchain-naver` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-naver

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


chat = ChatClovaX(
    model="HCX-005",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

"""
## Invocation

In addition to `invoke` below, `ChatClovaX` also supports batch, stream and their async functionalities.
"""
logger.info("## Invocation")

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Korean. Translate the user sentence.",
    ),
    ("human", "I love using NAVER AI."),
]

ai_msg = chat.invoke(messages)
ai_msg

logger.debug(ai_msg.content)

"""
## Chaining

We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}. Translate the user sentence.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | chat
chain.invoke(
    {
        "input_language": "English",
        "output_language": "Korean",
        "input": "I love using NAVER AI.",
    }
)

"""
## Streaming
"""
logger.info("## Streaming")

system = "You are a helpful assistant that can teach Korean pronunciation."
human = "Could you let me know how to say '{phrase}' in Korean?"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat

for chunk in chain.stream({"phrase": "Hi"}):
    logger.debug(chunk.content, end="", flush=True)

"""
## Tool calling

CLOVA Studio supports tool calling (also known as "[function calling](https://api.ncloud-docs.com/docs/clovastudio-chatcompletionsv3-fc)") that lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool. It is extremely useful for building tool-using chains and agents, and for getting structured outputs from models more generally.

**Note**: You should set `max_tokens` larger than 1024 to utilize the tool calling feature in CLOVA Studio. 

### ChatClovaX.bind_tools()

With `ChatClovaX.bind_tools`, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model. Under the hood these are converted to an Ollama-compatible tool schemas, which looks like:

```
{
    "name": "...",
    "description": "...",
    "parameters": {...}  # JSONSchema
}
```

and passed in every model invocation.
"""
logger.info("## Tool calling")


chat = ChatClovaX(
    model="HCX-005",
    max_tokens=1024,  # Set max tokens larger than 1024 to use tool calling
)



class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(
        ..., description="The city and province, e.g. Seongnam-si, Gyeonggi-do"
    )


chat_with_tools = chat.bind_tools([GetWeather])

ai_msg = chat_with_tools.invoke(
    "what is the weather like in Bundang-gu?",
)
ai_msg

"""
### AIMessage.tool_calls

Notice that the AIMessage has a `tool_calls` attribute. This contains in a standardized ToolCall format that is model-provider agnostic.
"""
logger.info("### AIMessage.tool_calls")

ai_msg.tool_calls

"""
## Structured Outputs

For supporting model(s), you can use the [Structured Outputs](https://api.ncloud-docs.com/docs/clovastudio-chatcompletionsv3-so) feature to force the model to generates responses in a specific structure, such as Pydantic model or TypedDict or JSON.

**Note**: Structured Outputs requires Thinking mode to be disabled. Set `thinking.effort` to `none`.
"""
logger.info("## Structured Outputs")


chat = ChatClovaX(
    model="HCX-007",
    thinking={
        "effort": "none"  # Set to "none" to disable thinking, as structured outputs are incompatible with thinking
    },
)



class Weather(BaseModel):
    """Virtual weather info to tell user."""

    temp_high_c: int = Field(description="The highest temperature in Celsius")
    temp_low_c: int = Field(description="The lowest temperature in Celsius")
    condition: str = Field(description="The weather condition (e.g., sunny, rainy)")
    precipitation_percent: Optional[int] = Field(
        default=None,
        description="The chance of precipitation in percent (optional, can be None)",
    )

"""
**Note**: CLOVA Studio supports Structured Outputs with a json schema method. Set `method` to `json_schema`.
"""

structured_chat = chat.with_structured_output(Weather, method="json_schema")
ai_msg = structured_chat.invoke(
    "what is the weather like in Bundang-gu?",
)
ai_msg

"""
## Thinking

For supporting model(s), when [Thinking](https://api.ncloud-docs.com/docs/clovastudio-chatcompletionsv3-thinking) feature is enabled (by default), it will output the step-by-step reasoning process that led to its final answer.

Specify the `thinking` parameter to control the feature—enable or disable the thinking process and configure its depth.
"""
logger.info("## Thinking")


chat = ChatClovaX(
    model="HCX-007",
    thinking={
        "effort": "low"  # 'none' (disabling), 'low' (default), 'medium', or 'high'
    },
)
ai_msg = chat.invoke("What is 3^3?")
logger.debug(ai_msg.content)

"""
### Accessing the thinking process

When Thinking mode is enabled, you can access the thinking process through the `thinking_content` attribute in `AIMessage.additional_kwargs`.
"""
logger.info("### Accessing the thinking process")

logger.debug(ai_msg.additional_kwargs["thinking_content"])

"""
## Additional functionalities

### Using fine-tuned models

You can call fine-tuned models by passing the `task_id` to the `model` parameter as: `ft:{task_id}`.

You can check `task_id` from corresponding Test App or Service App details.
"""
logger.info("## Additional functionalities")

fine_tuned_model = ChatClovaX(
    model="ft:a1b2c3d4",  # set as `ft:{task_id}` with your fine-tuned model's task id
)

fine_tuned_model.invoke(messages)

"""
## API reference

For detailed documentation of all ChatClovaX features and configurations head to the [API reference](https://guide.ncloud-docs.com/docs/clovastudio-dev-langchain)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
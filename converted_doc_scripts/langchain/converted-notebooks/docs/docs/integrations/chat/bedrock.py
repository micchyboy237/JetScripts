from jet.logger import logger
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
import os
import requests
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
sidebar_label: AWS Bedrock
---

# ChatBedrock

This doc will help you get started with AWS Bedrock [chat models](/docs/concepts/chat_models). Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Ollama, Cohere, Meta, Stability AI, and Amazon via a single API, along with a broad set of capabilities you need to build generative AI applications with security, privacy, and responsible AI. Using Amazon Bedrock, you can easily experiment with and evaluate top FMs for your use case, privately customize them with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), and build agents that execute tasks using your enterprise systems and data sources. Since Amazon Bedrock is serverless, you don't have to manage any infrastructure, and you can securely integrate and deploy generative AI capabilities into your applications using the AWS services you are already familiar with.

AWS Bedrock maintains a [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html) which provides a unified conversational interface for Bedrock models. This API does not yet support custom models. You can see a list of all [models that are supported here](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html).

:::info

We recommend the Converse API for users who do not need to use custom models. It can be accessed using [ChatBedrockConverse](https://python.langchain.com/api_reference/aws/chat_models/langchain_aws.chat_models.bedrock_converse.ChatBedrockConverse.html).

:::

For detailed documentation of all Bedrock features and configurations head to the [API reference](https://python.langchain.com/api_reference/aws/chat_models/langchain_aws.chat_models.bedrock_converse.ChatBedrockConverse.html).

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/bedrock) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatBedrock](https://python.langchain.com/api_reference/aws/chat_models/langchain_aws.chat_models.bedrock.ChatBedrock.html) | [langchain-aws](https://python.langchain.com/api_reference/aws/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-aws?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-aws?style=flat-square&label=%20) |
| [ChatBedrockConverse](https://python.langchain.com/api_reference/aws/chat_models/langchain_aws.chat_models.bedrock_converse.ChatBedrockConverse.html) | [langchain-aws](https://python.langchain.com/api_reference/aws/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-aws?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-aws?style=flat-square&label=%20) |

### Model features

The below apply to both `ChatBedrock` and `ChatBedrockConverse`.

| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |

## Setup

To access Bedrock models you'll need to create an AWS account, set up the Bedrock API service, get an access key ID and secret key, and install the `langchain-aws` integration package.

### Credentials

Head to the [AWS docs](https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html) to sign up to AWS and setup your credentials.

Alternatively, `ChatBedrockConverse` will read from the following environment variables by default:
"""
logger.info("# ChatBedrock")



"""
You'll also need to turn on model access for your account, which you can do by following [these instructions](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html).

To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("You'll also need to turn on model access for your account, which you can do by following [these instructions](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html).")



"""
### Installation

The LangChain Bedrock integration lives in the `langchain-aws` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-aws

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatBedrockConverse(
    model_id="anthropic.llama3.2-v1:0",
)

"""
## Invocation
"""
logger.info("## Invocation")

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg

logger.debug(ai_msg.content)

"""
### Streaming

Note that `ChatBedrockConverse` emits content blocks while streaming:
"""
logger.info("### Streaming")

for chunk in llm.stream(messages):
    logger.debug(chunk)

"""
You can filter to text using the [.text()](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.text) method on the output:
"""
logger.info("You can filter to text using the [.text()](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.text) method on the output:")

for chunk in llm.stream(messages):
    logger.debug(chunk.text(), end="|")

"""
## Extended Thinking 

This guide focuses on implementing Extended Thinking using AWS Bedrock with LangChain's `ChatBedrockConverse` integration.

### Supported Models

Extended Thinking is available for the following Claude models on AWS Bedrock:

| Model | Model ID |
|-------|----------|
| **Claude Opus 4** | `anthropic.claude-opus-4-20250514-v1:0` |
| **Claude Sonnet 4** | `anthropic.claude-sonnet-4-20250514-v1:0` |
| **Claude 3.7 Sonnet** | `us.anthropic.claude-3-7-sonnet-20250219-v1:0` |
"""
logger.info("## Extended Thinking")


llm = ChatBedrockConverse(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-west-2",
    max_tokens=4096,
    additional_model_request_fields={
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
)

ai_msg = llm.invoke(messages)
ai_msg

logger.debug(ai_msg.content)

"""
### How extended thinking works

When extended thinking is turned on, Claude creates thinking content blocks where it outputs its internal reasoning. Claude incorporates insights from this reasoning before crafting a final response. The API response will include thinking content blocks, followed by text content blocks.
"""
logger.info("### How extended thinking works")

next_messages = messages + [("ai", ai_msg.content), ("human", "I love AI")]
next_messages

ai_msg = llm.invoke(next_messages)
ai_msg

logger.debug(ai_msg.content)

"""
## Prompt caching

Bedrock supports [caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) of elements of your prompts, including messages and tools. This allows you to re-use large documents, instructions, [few-shot documents](/docs/concepts/few_shot_prompting/), and other data to reduce latency and costs.

:::note

Not all models support prompt caching. See supported models [here](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html#prompt-caching-models).

:::

To enable caching on an element of a prompt, mark its associated content block using the `cachePoint` key. See example below:
"""
logger.info("## Prompt caching")


llm = ChatBedrockConverse(model="us.anthropic.claude-3-7-sonnet-20250219-v1:0")

get_response = requests.get(
    "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"
)
readme = get_response.text

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's LangChain, according to its README?",
            },
            {
                "type": "text",
                "text": f"{readme}",
            },
            {
                "cachePoint": {"type": "default"},
            },
        ],
    },
]

response_1 = llm.invoke(messages)
response_2 = llm.invoke(messages)

usage_1 = response_1.usage_metadata["input_token_details"]
usage_2 = response_2.usage_metadata["input_token_details"]

logger.debug(f"First invocation:\n{usage_1}")
logger.debug(f"\nSecond:\n{usage_2}")

"""
## Chaining

We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate.from_messages(
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

For detailed documentation of all ChatBedrock features and configurations head to the API reference: https://python.langchain.com/api_reference/aws/chat_models/langchain_aws.chat_models.bedrock.ChatBedrock.html

For detailed documentation of all ChatBedrockConverse features and configurations head to the API reference: https://python.langchain.com/api_reference/aws/chat_models/langchain_aws.chat_models.bedrock_converse.ChatBedrockConverse.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
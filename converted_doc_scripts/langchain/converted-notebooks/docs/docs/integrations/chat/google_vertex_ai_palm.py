from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
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
sidebar_label: Google Cloud Vertex AI
---

# ChatVertexAI

This page provides a quick overview for getting started with VertexAI [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatVertexAI features and configurations head to the [API reference](https://python.langchain.com/api_reference/google_vertexai/chat_models/langchain_google_vertexai.chat_models.ChatVertexAI.html).

ChatVertexAI exposes all foundational models available in Google Cloud, like `gemini-2.5-pro`, `gemini-2.5-flash`, etc. For a full and updated list of available models visit [VertexAI documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models).

:::info Google Cloud VertexAI vs Google PaLM

The Google Cloud VertexAI integration is separate from the [Google PaLM integration](/docs/integrations/chat/google_generative_ai/). Google has chosen to offer an enterprise version of PaLM through GCP, and this supports the models made available through there.

:::

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/google_vertex_ai) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatVertexAI](https://python.langchain.com/api_reference/google_vertexai/chat_models/langchain_google_vertexai.chat_models.ChatVertexAI.html) | [langchain-google-vertexai](https://python.langchain.com/api_reference/google_vertexai/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-google-vertexai?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-google-vertexai?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

## Setup

To access VertexAI models you'll need to create a Google Cloud Platform account, set up credentials, and install the `langchain-google-vertexai` integration package.

### Credentials

To use the integration you must either:
- Have credentials configured for your environment (gcloud, workload identity, etc...)
- Store the path to a service account JSON file as the GOOGLE_APPLICATION_CREDENTIALS environment variable

This codebase uses the `google.auth` library which first looks for the application credentials variable mentioned above, and then looks for system-level auth.

For more information, see:
- https://cloud.google.com/docs/authentication/application-default-credentials#GAC
- https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth

To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("# ChatVertexAI")



"""
### Installation

The LangChain VertexAI integration lives in the `langchain-google-vertexai` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-google-vertexai

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatVertexAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    max_retries=6,
    stop=None,
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
## Built-in tools

Gemini supports a range of tools that are executed server-side.

### Google search

:::info Requires ``langchain-google-vertexai>=2.0.11``
:::

Gemini can execute a Google search and use the results to [ground its responses](https://ai.google.dev/gemini-api/docs/grounding):
"""
logger.info("## Built-in tools")


llm = ChatVertexAI(model="gemini-2.5-flash").bind_tools([{"google_search": {}}])

response = llm.invoke("What is today's news?")

"""
### Code execution

:::info Requires ``langchain-google-vertexai>=2.0.25``
:::

Gemini can [generate and execute Python code](https://ai.google.dev/gemini-api/docs/code-execution):
"""
logger.info("### Code execution")


llm = ChatVertexAI(model="gemini-2.5-flash").bind_tools([{"code_execution": {}}])

response = llm.invoke("What is 3^3?")

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

For detailed documentation of all ChatVertexAI features and configurations, like how to send multimodal inputs and configure safety settings, head to the API reference: https://python.langchain.com/api_reference/google_vertexai/chat_models/langchain_google_vertexai.chat_models.ChatVertexAI.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
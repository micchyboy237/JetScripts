from jet.adapters.langchain.chat_ollama import AzureChatOllama
from jet.logger import logger
from langchain_community.callbacks import get_ollama_callback
from langchain_core.prompts import ChatPromptTemplate
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
sidebar_label: Azure Ollama
---

# AzureChatOllama

This guide will help you get started with AzureOllama [chat models](/docs/concepts/chat_models). For detailed documentation of all AzureChatOllama features and configurations head to the [API reference](https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.azure.AzureChatOllama.html).

Azure Ollama has several chat models. You can find information about their latest models and their costs, context windows, and supported input types in the [Azure docs](https://learn.microsoft.com/en-us/azure/ai-services/ollama/concepts/models).

:::info Azure Ollama vs Ollama

Azure Ollama refers to Ollama models hosted on the [Microsoft Azure platform](https://azure.microsoft.com/en-us/products/ai-services/ollama-service). Ollama also provides its own model APIs. To access Ollama services directly, use the [ChatOllama integration](/docs/integrations/chat/ollama/).

:::

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/azure) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [AzureChatOllama](https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.azure.AzureChatOllama.html) | [langchain-ollama](https://python.langchain.com/api_reference/ollama/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-ollama?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-ollama?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |

## Setup

To access AzureOllama models you'll need to create an Azure account, create a deployment of an Azure Ollama model, get the name and endpoint for your deployment, get an Azure Ollama API key, and install the `langchain-ollama` integration package.

### Credentials

# Head to the [Azure docs](https://learn.microsoft.com/en-us/azure/ai-services/ollama/chatgpt-quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python) to create your deployment and generate an API key. Once you've done this set the AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables:
"""
logger.info("# AzureChatOllama")

# import getpass

# if "AZURE_OPENAI_API_KEY" not in os.environ:
#     os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass(
        "Enter your AzureOllama API key: "
    )
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://YOUR-ENDPOINT.ollama.azure.com/"

"""
T
o
 
e
n
a
b
l
e
 
a
u
t
o
m
a
t
e
d
 
t
r
a
c
i
n
g
 
o
f
 
y
o
u
r
 
m
o
d
e
l
 
c
a
l
l
s
,
 
s
e
t
 
y
o
u
r
 
[
L
a
n
g
S
m
i
t
h
]
(
h
t
t
p
s
:
/
/
d
o
c
s
.
s
m
i
t
h
.
l
a
n
g
c
h
a
i
n
.
c
o
m
/
)
 
A
P
I
 
k
e
y
:
"""
logger.info("T")



"""
### Installation

The LangChain AzureOllama integration lives in the `langchain-ollama` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-ollama

"""
## Instantiation

Now we can instantiate our model object and generate chat completions.
- Replace `azure_deployment` with the name of your deployment,
- You can find the latest supported `api_version` here: https://learn.microsoft.com/en-us/azure/ai-services/ollama/reference.
"""
logger.info("## Instantiation")


llm = AzureChatOllama(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
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
## Specifying model version

Azure Ollama responses contain `model_name` response metadata property, which is name of the model used to generate the response. However unlike native Ollama responses, it does not contain the specific version of the model, which is set on the deployment in Azure. E.g. it does not distinguish between `gpt-35-turbo-0125` and `gpt-35-turbo-0301`. This makes it tricky to know which version of the model was used to generate the response, which as result can lead to e.g. wrong total cost calculation with `OllamaCallbackHandler`.

To solve this problem, you can pass `model_version` parameter to `AzureChatOllama` class, which will be added to the model name in the llm output. This way you can easily distinguish between different versions of the model.
"""
logger.info("## Specifying model version")

# %pip install -qU langchain-community


with get_ollama_callback() as cb:
    llm.invoke(messages)
    logger.debug(
        f"Total Cost (USD): ${format(cb.total_cost, '.6f')}"
    )  # without specifying the model version, flat-rate 0.002 USD per 1k input and output tokens is used

llm_0301 = AzureChatOllama(
    azure_deployment="gpt-35-turbo",  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    model_version="0301",
)
with get_ollama_callback() as cb:
    llm_0301.invoke(messages)
    logger.debug(f"Total Cost (USD): ${format(cb.total_cost, '.6f')}")

"""
## API reference

For detailed documentation of all AzureChatOllama features and configurations head to the API reference: https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.azure.AzureChatOllama.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.prompts import PromptTemplate
import httpx
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
# Ollama

:::caution
You are currently on a page documenting the use of Ollama [text completion models](/docs/concepts/text_llms). The latest and most popular Ollama models are [chat completion models](/docs/concepts/chat_models).

Unless you are specifically using `gpt-3.5-turbo-instruct`, you are probably looking for [this page instead](/docs/integrations/chat/ollama/).
:::

[Ollama](https://platform.ollama.com/docs/introduction) offers a spectrum of models with different levels of power suitable for different tasks.

This example goes over how to use LangChain to interact with `Ollama` [models](https://platform.ollama.com/docs/models)

## Overview

### Integration details
| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/ollama) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatOllama](https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.base.ChatOllama.html) | [langchain-ollama](https://python.langchain.com/api_reference/ollama/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-ollama?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-ollama?style=flat-square&label=%20) |


## Setup

To access Ollama models you'll need to create an Ollama account, get an API key, and install the `langchain-ollama` integration package.

### Credentials

# Head to https://platform.ollama.com to sign up to Ollama and generate an API key. Once you've done this set the OPENAI_API_KEY environment variable:
"""
logger.info("# Ollama")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Ollama API key: ")

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

The LangChain Ollama integration lives in the `langchain-ollama` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-ollama

"""
Should you need to specify your organization ID, you can use the following cell. However, it is not required if you are only part of a single organization or intend to use your default organization. You can check your default organization [here](https://platform.ollama.com/account/api-keys).

To specify your organization, you can use this:
```python
# OPENAI_ORGANIZATION = getpass()

os.environ["OPENAI_ORGANIZATION"] = OPENAI_ORGANIZATION
```

## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatOllama()

"""
## Invocation
"""
logger.info("## Invocation")

llm.invoke("Hello how are you?")

"""
## Chaining
"""
logger.info("## Chaining")


prompt = PromptTemplate.from_template(
    "How to say {input} in {output_language}:\n")

chain = prompt | llm
chain.invoke(
    {
        "output_language": "German",
        "input": "I love programming.",
    }
)

"""
## Using a proxy

If you are behind an explicit proxy, you can specify the http_client to pass through
"""
logger.info("## Using a proxy")

# %pip install httpx


ollama = Ollama(
    model_name="gpt-3.5-turbo-instruct",
    http_client=httpx.Client(proxies="http://proxy.yourcompany.com:8080"),
)

"""
## API reference

For detailed documentation of all `Ollama` llm features and configurations head to the API reference: https://python.langchain.com/api_reference/ollama/llms/jet.adapters.langchain.chat_ollama.llms.base.Ollama.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)

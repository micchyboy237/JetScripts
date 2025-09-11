from __module_name__ import Chat__ModuleName__
from jet.logger import logger
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
sidebar_label: __ModuleName__
---

# Chat__ModuleName__

- TODO: Make sure API reference link is correct.

This will help you get started with __ModuleName__ [chat models](/docs/concepts/chat_models). For detailed documentation of all Chat__ModuleName__ features and configurations head to the [API reference](https://python.langchain.com/api_reference/__package_name_short_snake__/chat_models/__module_name__.chat_models.Chat__ModuleName__.html).

- TODO: Add any other relevant links, like information about models, prices, context windows, etc. See https://python.langchain.com/docs/integrations/chat/ollama/ for an example.

## Overview
### Integration details

- TODO: Fill in table features.
- TODO: Remove JS support link if not relevant, otherwise ensure link is correct.
- TODO: Make sure API reference links are correct.

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/__package_name_short_snake__) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [Chat__ModuleName__](https://python.langchain.com/api_reference/__package_name_short_snake__/chat_models/__module_name__.chat_models.Chat__ModuleName__.html) | [__package_name__](https://python.langchain.com/api_reference/__package_name_short_snake__/) | ✅/❌ | beta/❌ | ✅/❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/__package_name__?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/__package_name__?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ | ✅/❌ |

## Setup

- TODO: Update with relevant info.

To access __ModuleName__ models you'll need to create a/an __ModuleName__ account, get an API key, and install the `__package_name__` integration package.

### Credentials

- TODO: Update with relevant info.

Head to (TODO: link) to sign up to __ModuleName__ and generate an API key. Once you've done this set the __MODULE_NAME___API_KEY environment variable:
"""
logger.info("# Chat__ModuleName__")

# import getpass

if not os.getenv("__MODULE_NAME___API_KEY"):
#     os.environ["__MODULE_NAME___API_KEY"] = getpass.getpass("Enter your __ModuleName__ API key: ")

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

The LangChain __ModuleName__ integration lives in the `__package_name__` package:
"""
logger.info("### Installation")

# %pip install -qU __package_name__

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:

- TODO: Update model instantiation with relevant params.
"""
logger.info("## Instantiation")


llm = Chat__ModuleName__(
    model="model-name",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

"""
## Invocation

- TODO: Run cells so output can be seen.
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

- TODO: Run cells so output can be seen.
"""
logger.info("## Chaining")


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
## TODO: Any functionality specific to this model provider

E.g. creating/using finetuned models via this provider. Delete if not relevant.

## API reference

For detailed documentation of all Chat__ModuleName__ features and configurations head to the [API reference](https://python.langchain.com/api_reference/__package_name_short_snake__/chat_models/__module_name__.chat_models.Chat__ModuleName__.html)
"""
logger.info("## TODO: Any functionality specific to this model provider")

logger.info("\n\n[DONE]", bright=True)
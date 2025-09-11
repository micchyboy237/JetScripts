from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_together import ChatTogether
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
sidebar_label: Together
---

# ChatTogether


This page will help you get started with Together AI [chat models](../../concepts/chat_models.mdx). For detailed documentation of all ChatTogether features and configurations, head to the [API reference](https://python.langchain.com/api_reference/together/chat_models/langchain_together.chat_models.ChatTogether.html).

[Together AI](https://www.together.ai/) offers an API to query [50+ leading open-source models](https://docs.together.ai/docs/chat-models)

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/togetherai) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatTogether](https://python.langchain.com/api_reference/together/chat_models/langchain_together.chat_models.ChatTogether.html) | [langchain-together](https://python.langchain.com/api_reference/together/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-together?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-together?style=flat-square&label=%20) |

### Model features
| [Tool calling](../../how_to/tool_calling.ipynb) | [Structured output](../../how_to/structured_output.ipynb) | JSON mode | [Image input](../../how_to/multimodal_inputs.ipynb) | Audio input | Video input | [Token-level streaming](../../how_to/chat_streaming.ipynb) | Native async | [Token usage](../../how_to/chat_token_usage_tracking.ipynb) | [Logprobs](../../how_to/logprobs.ipynb) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |

## Setup

To access Together models you'll need to create a/an Together account, get an API key, and install the `langchain-together` integration package.

### Credentials

Head to [this page](https://api.together.ai) to sign up to Together and generate an API key. Once you've done this, set the TOGETHER_API_KEY environment variable:
"""
logger.info("# ChatTogether")

# import getpass

if "TOGETHER_API_KEY" not in os.environ:
#     os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter your Together API key: ")

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

The LangChain Together integration is included in the `langchain-together` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-together

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
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

We can [chain](../../how_to/sequence.ipynb) our model with a prompt template as follows:
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

For detailed documentation of all ChatTogether features and configurations, head to the API reference: https://python.langchain.com/api_reference/together/chat_models/langchain_together.chat_models.ChatTogether.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
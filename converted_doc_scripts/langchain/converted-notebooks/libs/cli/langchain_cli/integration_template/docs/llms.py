from __module_name__ import __ModuleName__LLM
from jet.logger import logger
from langchain_core.prompts import PromptTemplate
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

# __ModuleName__LLM

- [ ] TODO: Make sure API reference link is correct

This will help you get started with __ModuleName__ completion models (LLMs) using LangChain. For detailed documentation on `__ModuleName__LLM` features and configuration options, please refer to the [API reference](https://api.python.langchain.com/en/latest/llms/__module_name__.llms.__ModuleName__LLM.html).

## Overview
### Integration details

- TODO: Fill in table features.
- TODO: Remove JS support link if not relevant, otherwise ensure link is correct.
- TODO: Make sure API reference links are correct.

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/llms/__package_name_short_snake__) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [__ModuleName__LLM](https://api.python.langchain.com/en/latest/llms/__module_name__.llms.__ModuleName__LLM.html) | [__package_name__](https://api.python.langchain.com/en/latest/__package_name_short_snake___api_reference.html) | ✅/❌ | beta/❌ | ✅/❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/__package_name__?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/__package_name__?style=flat-square&label=%20) |

## Setup

- TODO: Update with relevant info.

To access __ModuleName__ models you'll need to create a/an __ModuleName__ account, get an API key, and install the `__package_name__` integration package.

### Credentials

- TODO: Update with relevant info.

Head to (TODO: link) to sign up to __ModuleName__ and generate an API key. Once you've done this set the __MODULE_NAME___API_KEY environment variable:
"""
logger.info("# __ModuleName__LLM")

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


llm = __ModuleName__LLM(
    model="model-name",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

"""
## Invocation

- [ ] TODO: Run cells so output can be seen.
"""
logger.info("## Invocation")

input_text = "__ModuleName__ is an AI company that "

completion = llm.invoke(input_text)
completion

"""
## Chaining

We can [chain](/docs/how_to/sequence/) our completion model with a prompt template like so:

- TODO: Run cells so output can be seen.
"""
logger.info("## Chaining")


prompt = PromptTemplate(
    "How to say {input} in {output_language}:\n"
)

chain = prompt | llm
chain.invoke(
    {
        "output_language": "German",
        "input": "I love programming.",
    }
)

"""
## TODO: Any functionality specific to this model provider

E.g. creating/using finetuned models via this provider. Delete if not relevant

## API reference

For detailed documentation of all `__ModuleName__LLM` features and configurations head to the API reference: https://api.python.langchain.com/en/latest/llms/__module_name__.llms.__ModuleName__LLM.html
"""
logger.info("## TODO: Any functionality specific to this model provider")

logger.info("\n\n[DONE]", bright=True)
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import ChatModelTabs from "@theme/ChatModelTabs";
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
title: Tagging
sidebar_class_name: hidden
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/tagging.ipynb)

# Classify Text into Labels

Tagging means labeling a document with classes such as:

- Sentiment
- Language
- Style (formal, informal etc.)
- Covered topics
- Political tendency

![Image description](../../static/img/tagging.png)

## Overview

Tagging has a few components:

* `function`: Like [extraction](/docs/tutorials/extraction), tagging uses [functions](https://ollama.com/blog/function-calling-and-other-api-updates) to specify how the model should tag a document
* `schema`: defines how we want to tag the document

## Quickstart

Let's see a very straightforward example of how we can use Ollama tool calling for tagging in LangChain. We'll use the [`with_structured_output`](/docs/how_to/structured_output) method supported by Ollama models.
"""
logger.info("# Classify Text into Labels")

pip install -U langchain-core

"""
We'll need to load a [chat model](/docs/integrations/chat/):


<ChatModelTabs customVarName="llm" />
"""
logger.info("We'll need to load a [chat model](/docs/integrations/chat/):")


llm = ChatOllama(model="llama3.2")

"""
Let's specify a Pydantic model with a few properties and their expected type in our schema.
"""
logger.info("Let's specify a Pydantic model with a few properties and their expected type in our schema.")


tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


structured_llm = llm.with_structured_output(Classification)

inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)

response

"""
If we want dictionary output, we can just call `.model_dump()`
"""
logger.info("If we want dictionary output, we can just call `.model_dump()`")

inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
prompt = tagging_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)

response.model_dump()

"""
As we can see in the examples, it correctly interprets what we want.

The results vary so that we may get, for example, sentiments in different languages ('positive', 'enojado' etc.).

We will see how to control these results in the next section.

## Finer control

Careful schema definition gives us more control over the model's output. 

Specifically, we can define:

- Possible values for each property
- Description to make sure that the model understands the property
- Required properties to be returned

Let's redeclare our Pydantic model to control for each of the previously mentioned aspects using enums:
"""
logger.info("## Finer control")

class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

llm = ChatOllama(model="llama3.2").with_structured_output(
    Classification
)

"""
Now the answers will be restricted in a way we expect!
"""
logger.info("Now the answers will be restricted in a way we expect!")

inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
llm.invoke(prompt)

inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
prompt = tagging_prompt.invoke({"input": inp})
llm.invoke(prompt)

inp = "Weather is ok here, I can go outside without much more than a coat"
prompt = tagging_prompt.invoke({"input": inp})
llm.invoke(prompt)

"""
The [LangSmith trace](https://smith.langchain.com/public/38294e04-33d8-4c5a-ae92-c2fe68be8332/r) lets us peek under the hood:

![Image description](../../static/img/tagging_trace.png)

### Going deeper

* You can use the [metadata tagger](/docs/integrations/document_transformers/ollama_metadata_tagger) document transformer to extract metadata from a LangChain `Document`. 
* This covers the same basic functionality as the tagging chain, only applied to a LangChain `Document`.
"""
logger.info("### Going deeper")

logger.info("\n\n[DONE]", bright=True)
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import tool_example_to_messages
from pydantic import BaseModel, Field
from typing import List, Optional
from typing import Optional
import ChatModelTabs from "@theme/ChatModelTabs";
import CodeBlock from "@theme/CodeBlock";
import TabItem from '@theme/TabItem';
import Tabs from '@theme/Tabs';
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
sidebar_position: 4
---

# Build an Extraction Chain

In this tutorial, we will use [tool-calling](/docs/concepts/tool_calling) features of [chat models](/docs/concepts/chat_models) to extract structured information from unstructured text. We will also demonstrate how to use [few-shot prompting](/docs/concepts/few_shot_prompting/) in this context to improve performance.

:::important
This tutorial requires `langchain-core>=0.3.20` and will only work with models that support **tool calling**.
:::

## Setup

### Jupyter Notebook

This and other tutorials are perhaps most conveniently run in a [Jupyter notebooks](https://jupyter.org/). Going through guides in an interactive environment is a great way to better understand them. See [here](https://jupyter.org/install) for instructions on how to install.

### Installation

To install LangChain run:


<Tabs>
  <TabItem value="pip" label="Pip" default>
    <CodeBlock language="bash">pip install --upgrade langchain-core</CodeBlock>
  </TabItem>
  <TabItem value="conda" label="Conda">
    <CodeBlock language="bash">conda install langchain-core -c conda-forge</CodeBlock>
  </TabItem>
</Tabs>



For more details, see our [Installation guide](/docs/how_to/installation).

### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls.
As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent.
The best way to do this is with [LangSmith](https://smith.langchain.com).

After you sign up at the link above, make sure to set your environment variables to start logging traces:

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

Or, if in a notebook, you can set them with:

```python
# import getpass

os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

## The Schema

First, we need to describe what information we want to extract from the text.

We'll use Pydantic to define an example schema  to extract personal information.
"""
logger.info("# Build an Extraction Chain")




class Person(BaseModel):
    """Information about a person."""


    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )

"""
There are two best practices when defining schema:

1. Document the **attributes** and the **schema** itself: This information is sent to the LLM and is used to improve the quality of information extraction.
2. Do not force the LLM to make up information! Above we used `Optional` for the attributes allowing the LLM to output `None` if it doesn't know the answer.

:::important
For best performance, document the schema well and make sure the model isn't forced to return results if there's no information to be extracted in the text.
:::

## The Extractor

Let's create an information extractor using the schema we defined above.
"""
logger.info("## The Extractor")


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

"""
We need to use a model that supports function/tool calling.

Please review [the documentation](/docs/concepts/tool_calling) for all models that can be used with this API.


<ChatModelTabs customVarName="llm" />
"""
logger.info("We need to use a model that supports function/tool calling.")


llm = ChatOllama(model="llama3.2")

structured_llm = llm.with_structured_output(schema=Person)

"""
Let's test it out:
"""
logger.info("Let's test it out:")

text = "Alan Smith is 6 feet tall and has blond hair."
prompt = prompt_template.invoke({"text": text})
structured_llm.invoke(prompt)

"""
:::important 

Extraction is Generative 🤯

LLMs are generative models, so they can do some pretty cool things like correctly extract the height of the person in meters
even though it was provided in feet!
:::

We can see the LangSmith trace [here](https://smith.langchain.com/public/44b69a63-3b3b-47b8-8a6d-61b46533f015/r). Note that the [chat model portion of the trace](https://smith.langchain.com/public/44b69a63-3b3b-47b8-8a6d-61b46533f015/r/dd1f6305-f1e9-4919-bd8f-339d03a12d01) reveals the exact sequence of messages sent to the model, tools invoked, and other metadata.

## Multiple Entities

In **most cases**, you should be extracting a list of entities rather than a single entity.

This can be easily achieved using pydantic by nesting models inside one another.
"""
logger.info("## Multiple Entities")




class Person(BaseModel):
    """Information about a person."""


    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


class Data(BaseModel):
    """Extracted data about people."""

    people: List[Person]

"""
:::important
Extraction results might not be perfect here. Read on to see how to use **Reference Examples** to improve the quality of extraction, and check out our extraction [how-to](/docs/how_to/#extraction) guides for more detail.
:::
"""
logger.info("Extraction results might not be perfect here. Read on to see how to use **Reference Examples** to improve the quality of extraction, and check out our extraction [how-to](/docs/how_to/#extraction) guides for more detail.")

structured_llm = llm.with_structured_output(schema=Data)
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt = prompt_template.invoke({"text": text})
structured_llm.invoke(prompt)

"""
:::tip
When the schema accommodates the extraction of **multiple entities**, it also allows the model to extract **no entities** if no relevant information
is in the text by providing an empty list. 

This is usually a **good** thing! It allows specifying **required** attributes on an entity without necessarily forcing the model to detect this entity.
:::

We can see the LangSmith trace [here](https://smith.langchain.com/public/7173764d-5e76-45fe-8496-84460bd9cdef/r).

## Reference examples

The behavior of LLM applications can be steered using [few-shot prompting](/docs/concepts/few_shot_prompting/). For [chat models](/docs/concepts/chat_models/), this can take the form of a sequence of pairs of input and response messages demonstrating desired behaviors.

For example, we can convey the meaning of a symbol with alternating `user` and `assistant` [messages](/docs/concepts/messages/#role):
"""
logger.info("## Reference examples")

messages = [
    {"role": "user", "content": "2 🦜 2"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "2 🦜 3"},
    {"role": "assistant", "content": "5"},
    {"role": "user", "content": "3 🦜 4"},
]

response = llm.invoke(messages)
logger.debug(response.content)

"""
[Structured output](/docs/concepts/structured_outputs/) often uses [tool calling](/docs/concepts/tool_calling/) under-the-hood. This typically involves the generation of [AI messages](/docs/concepts/messages/#aimessage) containing tool calls, as well as [tool messages](/docs/concepts/messages/#toolmessage) containing the results of tool calls. What should a sequence of messages look like in this case?

Different [chat model providers](/docs/integrations/chat/) impose different requirements for valid message sequences. Some will accept a (repeating) message sequence of the form:

- User message
- AI message with tool call
- Tool message with result

Others require a final AI message containing some sort of response.

LangChain includes a utility function [tool_example_to_messages](https://python.langchain.com/api_reference/core/utils/langchain_core.utils.function_calling.tool_example_to_messages.html) that will generate a valid sequence for most model providers. It simplifies the generation of structured few-shot examples by just requiring Pydantic representations of the corresponding tool calls.

Let's try this out. We can convert pairs of input strings and desired Pydantic objects to a sequence of messages that can be provided to a chat model. Under the hood, LangChain will format the tool calls to each provider's required format.

Note: this version of `tool_example_to_messages` requires `langchain-core>=0.3.20`.
"""
logger.info("Different [chat model providers](/docs/integrations/chat/) impose different requirements for valid message sequences. Some will accept a (repeating) message sequence of the form:")


examples = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep.",
        Data(people=[]),
    ),
    (
        "Fiona traveled far from France to Spain.",
        Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
    ),
]


messages = []

for txt, tool_call in examples:
    if tool_call.people:
        ai_response = "Detected people."
    else:
        ai_response = "Detected no people."
    messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

"""
Inspecting the result, we see these two example pairs generated eight messages:
"""
logger.info("Inspecting the result, we see these two example pairs generated eight messages:")

for message in messages:
    message.pretty_logger.debug()

"""
Let's compare performance with and without these messages. For example, let's pass a message for which we intend no people to be extracted:
"""
logger.info("Let's compare performance with and without these messages. For example, let's pass a message for which we intend no people to be extracted:")

message_no_extraction = {
    "role": "user",
    "content": "The solar system is large, but earth has only 1 moon.",
}

structured_llm = llm.with_structured_output(schema=Data)
structured_llm.invoke([message_no_extraction])

"""
In this example, the model is liable to erroneously generate records of people.

Because our few-shot examples contain examples of "negatives", we encourage the model to behave correctly in this case:
"""
logger.info("In this example, the model is liable to erroneously generate records of people.")

structured_llm.invoke(messages + [message_no_extraction])

"""
:::tip

The [LangSmith](https://smith.langchain.com/public/b3433f57-7905-4430-923c-fed214525bf1/r) trace for the run reveals the exact sequence of messages sent to the chat model, tool calls generated, latency, token counts, and other metadata.

:::

See [this guide](/docs/how_to/extraction_examples/) for more detail on extraction workflows with reference examples, including how to incorporate prompt templates and customize the generation of example messages.

## Next steps

Now that you understand the basics of extraction with LangChain, you're ready to proceed to the rest of the how-to guides:

- [Add Examples](/docs/how_to/extraction_examples): More detail on using **reference examples** to improve performance.
- [Handle Long Text](/docs/how_to/extraction_long_text): What should you do if the text does not fit into the context window of the LLM?
- [Use a Parsing Approach](/docs/how_to/extraction_parse): Use a prompt based approach to extract with models that do not support **tool/function calling**.
"""
logger.info("## Next steps")


logger.info("\n\n[DONE]", bright=True)
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import (
AIMessage,
BaseMessage,
HumanMessage,
SystemMessage,
ToolMessage,
)
from langchain_core.messages import (
HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import Dict, List, TypedDict
from typing import List, Optional
import ChatModelTabs from "@theme/ChatModelTabs";
import os
import shutil
import uuid


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
# How to use reference examples when doing extraction

The quality of extractions can often be improved by providing reference examples to the LLM.

Data extraction attempts to generate [structured representations](/docs/concepts/structured_outputs/) of information found in text and other unstructured or semi-structured formats. [Tool-calling](/docs/concepts/tool_calling) LLM features are often used in this context. This guide demonstrates how to build few-shot examples of tool calls to help steer the behavior of extraction and similar applications.

:::tip
While this guide focuses how to use examples with a tool calling model, this technique is generally applicable, and will work
also with JSON more or prompt based techniques.
:::

LangChain implements a [tool-call attribute](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.tool_calls) on messages from LLMs that include tool calls. See our [how-to guide on tool calling](/docs/how_to/tool_calling) for more detail. To build reference examples for data extraction, we build a chat history containing a sequence of: 

- [HumanMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.human.HumanMessage.html) containing example inputs;
- [AIMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html) containing example tool calls;
- [ToolMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.tool.ToolMessage.html) containing example tool outputs.

LangChain adopts this convention for structuring tool calls into conversation across LLM model providers.

First we build a prompt template that includes a placeholder for these messages:
"""
logger.info("# How to use reference examples when doing extraction")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked "
            "to extract, return null for the attribute's value.",
        ),
        MessagesPlaceholder("examples"),  # <-- EXAMPLES!
        ("human", "{text}"),
    ]
)

"""
Test out the template:
"""
logger.info("Test out the template:")


prompt.invoke(
    {"text": "this is some text", "examples": [HumanMessage(content="testing 1 2 3")]}
)

"""
## Define the schema

Let's re-use the person schema from the [extraction tutorial](/docs/tutorials/extraction).
"""
logger.info("## Define the schema")




class Person(BaseModel):
    """Information about a person."""


    name: Optional[str] = Field(..., description="The name of the person")
    hair_color: Optional[str] = Field(
        ..., description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(..., description="Height in METERs")


class Data(BaseModel):
    """Extracted data about people."""

    people: List[Person]

"""
## Define reference examples

Examples can be defined as a list of input-output pairs. 

Each example contains an example `input` text and an example `output` showing what should be extracted from the text.

:::important
This is a bit in the weeds, so feel free to skip.

The format of the example needs to match the API used (e.g., tool calling or JSON mode etc.).

Here, the formatted examples will match the format expected for the tool calling API since that's what we're using.
:::
"""
logger.info("## Define reference examples")




class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    tool_calls = []
    for tool_call in example["tool_calls"]:
        tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "args": tool_call.dict(),
                "name": tool_call.__class__.__name__,
            },
        )
    messages.append(AIMessage(content="", tool_calls=tool_calls))
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(tool_calls)
    for output, tool_call in zip(tool_outputs, tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages

"""
Next let's define our examples and then convert them into message format.
"""
logger.info("Next let's define our examples and then convert them into message format.")

examples = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep. There are many fish in it.",
        Data(people=[]),
    ),
    (
        "Fiona traveled far from France to Spain.",
        Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
    ),
]


messages = []

for text, tool_call in examples:
    messages.extend(
        tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
    )

"""
Let's test out the prompt
"""
logger.info("Let's test out the prompt")

example_prompt = prompt.invoke({"text": "this is some text", "examples": messages})

for message in example_prompt.messages:
    logger.debug(f"{message.type}: {message}")

"""
## Create an extractor

Let's select an LLM. Because we are using tool-calling, we will need a model that supports a tool-calling feature. See [this table](/docs/integrations/chat) for available LLMs.


<ChatModelTabs
  customVarName="llm"
  overrideParams={{ollama: {model: "gpt-4-0125-preview", kwargs: "temperature=0"}}}
/>
"""
logger.info("## Create an extractor")


llm = ChatOllama(model="llama3.2")

"""
Following the [extraction tutorial](/docs/tutorials/extraction), we use the `.with_structured_output` method to structure model outputs according to the desired schema:
"""
logger.info("Following the [extraction tutorial](/docs/tutorials/extraction), we use the `.with_structured_output` method to structure model outputs according to the desired schema:")

runnable = prompt | llm.with_structured_output(
    schema=Data,
    method="function_calling",
    include_raw=False,
)

"""
## Without examples 😿

Notice that even capable models can fail with a **very simple** test case!
"""
logger.info("## Without examples 😿")

for _ in range(5):
    text = "The solar system is large, but earth has only 1 moon."
    logger.debug(runnable.invoke({"text": text, "examples": []}))

"""
## With examples 😻

Reference examples helps to fix the failure!
"""
logger.info("## With examples 😻")

for _ in range(5):
    text = "The solar system is large, but earth has only 1 moon."
    logger.debug(runnable.invoke({"text": text, "examples": messages}))

"""
Note that we can see the few-shot examples as tool-calls in the [Langsmith trace](https://smith.langchain.com/public/4c436bc2-a1ce-440b-82f5-093947542e40/r).

And we retain performance on a positive sample:
"""
logger.info("Note that we can see the few-shot examples as tool-calls in the [Langsmith trace](https://smith.langchain.com/public/4c436bc2-a1ce-440b-82f5-093947542e40/r).")

runnable.invoke(
    {
        "text": "My name is Harrison. My hair is black.",
        "examples": messages,
    }
)

logger.info("\n\n[DONE]", bright=True)
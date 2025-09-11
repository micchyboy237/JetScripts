from jet.adapters.langchain.chat_ollama.chat_models import ChatOllama
from jet.logger import logger
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import ChatModelTabs from "@theme/ChatModelTabs";
import json
import os
import re
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
# How to use prompting alone (no tool calling) to do extraction

[Tool calling](/docs/concepts/tool_calling/) features are not required for generating structured output from LLMs. LLMs that are able to follow prompt instructions well can be tasked with outputting information in a given format.

This approach relies on designing good prompts and then parsing the output of the LLMs to make them extract information well.

To extract data without tool-calling features: 

1. Instruct the LLM to generate text following an expected format (e.g., JSON with a certain schema);
2. Use [output parsers](/docs/concepts/output_parsers) to structure the model response into a desired Python object.

First we select a LLM:


<ChatModelTabs customVarName="model" />
"""
logger.info("# How to use prompting alone (no tool calling) to do extraction")


model = ChatOllama(model="llama3.2")

"""
:::tip
This tutorial is meant to be simple, but generally should really include reference examples to squeeze out performance!
:::

## Using PydanticOutputParser

The following example uses the built-in `PydanticOutputParser` to parse the output of a chat model.
"""
logger.info("## Using PydanticOutputParser")




class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )


class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]


parser = PydanticOutputParser(pydantic_object=People)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

"""
Let's take a look at what information is sent to the model
"""
logger.info("Let's take a look at what information is sent to the model")

query = "Anna is 23 years old and she is 6 feet tall"

logger.debug(prompt.format_prompt(query=query).to_string())

"""
Having defined our prompt, we simply chain together the prompt, model and output parser:
"""
logger.info("Having defined our prompt, we simply chain together the prompt, model and output parser:")

chain = prompt | model | parser
chain.invoke({"query": query})

"""
Check out the associated [Langsmith trace](https://smith.langchain.com/public/92ed52a3-92b9-45af-a663-0a9c00e5e396/r).

Note that the schema shows up in two places: 

1. In the prompt, via `parser.get_format_instructions()`;
2. In the chain, to receive the formatted output and structure it into a Python object (in this case, the Pydantic object `People`).

## Custom Parsing

If desired, it's easy to create a custom prompt and parser with `LangChain` and `LCEL`.

To create a custom parser, define a function to parse the output from the model (typically an [AIMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html)) into an object of your choice.

See below for a simple implementation of a JSON parser.
"""
logger.info("## Custom Parsing")




class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )


class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Output your answer as JSON that  "
            "matches the given schema: ```json\n{schema}\n```. "
            "Make sure to wrap the answer in ```json and ``` tags",
        ),
        ("human", "{query}"),
    ]
).partial(schema=People.schema())


def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    pattern = r"```json(.*?)```"

    matches = re.findall(pattern, text, re.DOTALL)

    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")

query = "Anna is 23 years old and she is 6 feet tall"
logger.debug(prompt.format_prompt(query=query).to_string())

chain = prompt | model | extract_json
chain.invoke({"query": query})

"""
## Other Libraries

If you're looking at extracting using a parsing approach, check out the [Kor](https://eyurtsev.github.io/kor/) library. It's written by one of the `LangChain` maintainers and it
helps to craft a prompt that takes examples into account, allows controlling formats (e.g., JSON or CSV) and expresses the schema in TypeScript. It seems to work pretty!
"""
logger.info("## Other Libraries")

logger.info("\n\n[DONE]", bright=True)
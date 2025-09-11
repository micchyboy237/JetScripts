from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, model_validator
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
sidebar_position: 3
---

# How to use output parsers to parse an LLM response into structured format

Language models output text. But there are times where you want to get more structured information than just text back. While some model providers support [built-in ways to return structured output](/docs/how_to/structured_output), not all do.

[Output parsers](/docs/concepts/output_parsers/) are classes that help structure language model responses. There are two main methods an output parser must implement:

- "Get format instructions": A method which returns a string containing instructions for how the output of a language model should be formatted.
- "Parse": A method which takes in a string (assumed to be the response from a language model) and parses it into some structure.

And then one optional one:

- "Parse with prompt": A method which takes in a string (assumed to be the response from a language model) and a prompt (assumed to be the prompt that generated such a response) and parses it into some structure. The prompt is largely provided in the event the OutputParser wants to retry or fix the output in some way, and needs information from the prompt to do so.

## Get started

Below we go over the main type of output parser, the `PydanticOutputParser`.
"""
logger.info(
    "# How to use output parsers to parse an LLM response into structured format")


model = Ollama(model_name="gpt-3.5-turbo-instruct", temperature=0.0)


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    @model_validator(mode="before")
    @classmethod
    def question_ends_with_question_mark(cls, values: dict) -> dict:
        setup = values.get("setup")
        if setup and setup[-1] != "?":
            raise ValueError("Badly formed question!")
        return values


parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()},
)

prompt_and_model = prompt | model
output = prompt_and_model.invoke({"query": "Tell me a joke."})
parser.invoke(output)

"""
## LCEL

Output parsers implement the [Runnable interface](/docs/concepts/runnables), the basic building block of the [LangChain Expression Language (LCEL)](/docs/concepts/lcel). This means they support `invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch`, `astream_log` calls.

Output parsers accept a string or `BaseMessage` as input and can return an arbitrary type.
"""
logger.info("## LCEL")

parser.invoke(output)

"""
Instead of manually invoking the parser, we also could've just added it to our `Runnable` sequence:
"""
logger.info(
    "Instead of manually invoking the parser, we also could've just added it to our `Runnable` sequence:")

chain = prompt | model | parser
chain.invoke({"query": "Tell me a joke."})

"""
While all parsers support the streaming interface, only certain parsers can stream through partially parsed objects, since this is highly dependent on the output type. Parsers which cannot construct partial objects will simply yield the fully parsed output.

The `SimpleJsonOutputParser` for example can stream through partial outputs:
"""
logger.info("While all parsers support the streaming interface, only certain parsers can stream through partially parsed objects, since this is highly dependent on the output type. Parsers which cannot construct partial objects will simply yield the fully parsed output.")


json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)
json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | model | json_parser

list(json_chain.stream({"question": "Who invented the microscope?"}))

"""
Similarly,for `PydanticOutputParser`:
"""
logger.info("Similarly,for `PydanticOutputParser`:")

list(chain.stream({"query": "Tell me a joke."}))

logger.info("\n\n[DONE]", bright=True)

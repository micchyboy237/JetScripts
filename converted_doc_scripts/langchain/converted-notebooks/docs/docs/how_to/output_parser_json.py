from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
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
# How to parse JSON output

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models)
- [Output parsers](/docs/concepts/output_parsers)
- [Prompt templates](/docs/concepts/prompt_templates)
- [Structured output](/docs/how_to/structured_output)
- [Chaining runnables together](/docs/how_to/sequence/)

:::

While some model providers support [built-in ways to return structured output](/docs/how_to/structured_output), not all do. We can use an output parser to help users to specify an arbitrary JSON schema via the prompt, query a model for outputs that conform to that schema, and finally parse that schema as JSON.

:::note
Keep in mind that large language models are leaky abstractions! You'll have to use an LLM with sufficient capacity to generate well-formed JSON.
:::

The [`JsonOutputParser`](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.json.JsonOutputParser.html) is one built-in option for prompting for and then parsing JSON output. While it is similar in functionality to the [`PydanticOutputParser`](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.pydantic.PydanticOutputParser.html), it also supports streaming back partial JSON objects.

Here's an example of how it can be used alongside [Pydantic](https://docs.pydantic.dev/) to conveniently declare the expected schema:
"""
logger.info("# How to parse JSON output")

# %pip install -qU langchain langchain-ollama

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()


model = ChatOllama(model="llama3.2")


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


joke_query = "Tell me a joke."

parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})

"""
Note that we are passing `format_instructions` from the parser directly into the prompt. You can and should experiment with adding your own formatting hints in the other parts of your prompt to either augment or replace the default instructions:
"""
logger.info("Note that we are passing `format_instructions` from the parser directly into the prompt. You can and should experiment with adding your own formatting hints in the other parts of your prompt to either augment or replace the default instructions:")

parser.get_format_instructions()

"""
## Streaming

As mentioned above, a key difference between the `JsonOutputParser` and the `PydanticOutputParser` is that the `JsonOutputParser` output parser supports streaming partial chunks. Here's what that looks like:
"""
logger.info("## Streaming")

for s in chain.stream({"query": joke_query}):
    logger.debug(s)

"""
## Without Pydantic

You can also use the `JsonOutputParser` without Pydantic. This will prompt the model to return JSON, but doesn't provide specifics about what the schema should be.
"""
logger.info("## Without Pydantic")

joke_query = "Tell me a joke."

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})

"""
## Next steps

You've now learned one way to prompt a model to return structured JSON. Next, check out the [broader guide on obtaining structured output](/docs/how_to/structured_output) for other techniques.
"""
logger.info("## Next steps")


logger.info("\n\n[DONE]", bright=True)
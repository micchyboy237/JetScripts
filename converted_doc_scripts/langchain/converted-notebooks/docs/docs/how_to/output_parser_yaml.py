from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.output_parsers import YamlOutputParser
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
# How to parse YAML output

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models)
- [Output parsers](/docs/concepts/output_parsers)
- [Prompt templates](/docs/concepts/prompt_templates)
- [Structured output](/docs/how_to/structured_output)
- [Chaining runnables together](/docs/how_to/sequence/)

:::

LLMs from different providers often have different strengths depending on the specific data they are trained on. This also means that some may be "better" and more reliable at generating output in formats other than JSON.

This output parser allows users to specify an arbitrary schema and query LLMs for outputs that conform to that schema, using YAML to format their response.

:::note
Keep in mind that large language models are leaky abstractions! You'll have to use an LLM with sufficient capacity to generate well-formed YAML.
:::
"""
logger.info("# How to parse YAML output")

# %pip install -qU langchain langchain-ollama

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

"""
We use [Pydantic](https://docs.pydantic.dev) with the [`YamlOutputParser`](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.yaml.YamlOutputParser.html#langchain.output_parsers.yaml.YamlOutputParser) to declare our data model and give the model more context as to what type of YAML it should generate:
"""
logger.info("We use [Pydantic](https://docs.pydantic.dev) with the [`YamlOutputParser`](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.yaml.YamlOutputParser.html#langchain.output_parsers.yaml.YamlOutputParser) to declare our data model and give the model more context as to what type of YAML it should generate:")



class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


model = ChatOllama(model="llama3.2")

joke_query = "Tell me a joke."

parser = YamlOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})

"""
The parser will automatically parse the output YAML and create a Pydantic model with the data. We can see the parser's `format_instructions`, which get added to the prompt:
"""
logger.info("The parser will automatically parse the output YAML and create a Pydantic model with the data. We can see the parser's `format_instructions`, which get added to the prompt:")

parser.get_format_instructions()

"""
You can and should experiment with adding your own formatting hints in the other parts of your prompt to either augment or replace the default instructions.

## Next steps

You've now learned how to prompt a model to return YAML. Next, check out the [broader guide on obtaining structured output](/docs/how_to/structured_output) for other related techniques.
"""
logger.info("## Next steps")


logger.info("\n\n[DONE]", bright=True)
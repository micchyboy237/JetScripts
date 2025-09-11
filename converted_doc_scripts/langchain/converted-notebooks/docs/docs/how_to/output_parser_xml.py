from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.output_parsers import XMLOutputParser
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
# How to parse XML output

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models)
- [Output parsers](/docs/concepts/output_parsers)
- [Prompt templates](/docs/concepts/prompt_templates)
- [Structured output](/docs/how_to/structured_output)
- [Chaining runnables together](/docs/how_to/sequence/)

:::

LLMs from different providers often have different strengths depending on the specific data they are trained on. This also means that some may be "better" and more reliable at generating output in formats other than JSON.

This guide shows you how to use the [`XMLOutputParser`](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.xml.XMLOutputParser.html) to prompt models for XML output, then and [parse](/docs/concepts/output_parsers/) that output into a usable format.

:::note
Keep in mind that large language models are leaky abstractions! You'll have to use an LLM with sufficient capacity to generate well-formed XML.
:::

In the following examples, we use Ollama's Claude-2 model (https://docs.anthropic.com/claude/docs), which is one such model that is optimized for XML tags.
"""
logger.info("# How to parse XML output")

# %pip install -qU langchain langchain-anthropic

# from getpass import getpass

# if "ANTHROPIC_API_KEY" not in os.environ:
#     os.environ["ANTHROPIC_API_KEY"] = getpass()

"""
Let's start with a simple request to the model.
"""
logger.info("Let's start with a simple request to the model.")


model = ChatOllama(model="llama3.2")

actor_query = "Generate the shortened filmography for Tom Hanks."

output = model.invoke(
    f"""{actor_query}
Please enclose the movies in <movie></movie> tags"""
)

logger.debug(output.content)

"""
This actually worked pretty well! But it would be nice to parse that XML into a more easily usable format. We can use the `XMLOutputParser` to both add default format instructions to the prompt and parse outputted XML into a dict:
"""
logger.info("This actually worked pretty well! But it would be nice to parse that XML into a more easily usable format. We can use the `XMLOutputParser` to both add default format instructions to the prompt and parse outputted XML into a dict:")

parser = XMLOutputParser()

parser.get_format_instructions()

prompt = PromptTemplate(
    template="""{query}\n{format_instructions}""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

output = chain.invoke({"query": actor_query})
logger.debug(output)

"""
We can also add some tags to tailor the output to our needs. You can and should experiment with adding your own formatting hints in the other parts of your prompt to either augment or replace the default instructions:
"""
logger.info("We can also add some tags to tailor the output to our needs. You can and should experiment with adding your own formatting hints in the other parts of your prompt to either augment or replace the default instructions:")

parser = XMLOutputParser(tags=["movies", "actor", "film", "name", "genre"])

parser.get_format_instructions()

prompt = PromptTemplate(
    template="""{query}\n{format_instructions}""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


chain = prompt | model | parser

output = chain.invoke({"query": actor_query})

logger.debug(output)

"""
This output parser also supports streaming of partial chunks. Here's an example:
"""
logger.info("This output parser also supports streaming of partial chunks. Here's an example:")

for s in chain.stream({"query": actor_query}):
    logger.debug(s)

"""
## Next steps

You've now learned how to prompt a model to return XML. Next, check out the [broader guide on obtaining structured output](/docs/how_to/structured_output) for other related techniques.
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)
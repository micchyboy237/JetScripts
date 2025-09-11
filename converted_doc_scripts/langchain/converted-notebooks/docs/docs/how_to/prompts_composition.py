from jet.logger import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate
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
sidebar_position: 5
---

# How to compose prompts together

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Prompt templates](/docs/concepts/prompt_templates)

:::

LangChain provides a user friendly interface for composing different parts of [prompts](/docs/concepts/prompt_templates/) together. You can do this with either string prompts or chat prompts. Constructing prompts this way allows for easy reuse of components.

## String prompt composition

When working with string prompts, each template is joined together. You can work with either prompts directly or strings (the first element in the list needs to be a prompt).
"""
logger.info("# How to compose prompts together")


prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)

prompt

prompt.format(topic="sports", language="spanish")

"""
## Chat prompt composition

A chat prompt is made up of a list of messages. Similarly to the above example, we can concatenate chat prompt templates. Each new element is a new message in the final prompt.

First, let's initialize the a [`ChatPromptTemplate`](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html) with a [`SystemMessage`](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.system.SystemMessage.html).
"""
logger.info("## Chat prompt composition")


prompt = SystemMessage(content="You are a nice pirate")

"""
You can then easily create a pipeline combining it with other messages *or* message templates.
Use a `Message` when there is no variables to be formatted, use a `MessageTemplate` when there are variables to be formatted. You can also use just a string (note: this will automatically get inferred as a [`HumanMessagePromptTemplate`](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.HumanMessagePromptTemplate.html).)
"""
logger.info("You can then easily create a pipeline combining it with other messages *or* message templates.")

new_prompt = (
    prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)

"""
Under the hood, this creates an instance of the ChatPromptTemplate class, so you can use it just as you did before!
"""
logger.info("Under the hood, this creates an instance of the ChatPromptTemplate class, so you can use it just as you did before!")

new_prompt.format_messages(input="i said hi")

"""
## Using PipelinePrompt

:::warning Deprecated

PipelinePromptTemplate is deprecated; for more information, please refer to [PipelinePromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.pipeline.PipelinePromptTemplate.html).

:::

LangChain includes a class called [`PipelinePromptTemplate`](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.pipeline.PipelinePromptTemplate.html), which can be useful when you want to reuse parts of prompts. A PipelinePrompt consists of two main parts:

- Final prompt: The final prompt that is returned
- Pipeline prompts: A list of tuples, consisting of a string name and a prompt template. Each prompt template will be formatted and then passed to future prompt templates as a variable with the same name.
"""
logger.info("## Using PipelinePrompt")


full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

introduction_template = """You are impersonating {person}."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

example_template = """Here's an example of an interaction:

Q: {example_q}
A: {example_a}"""
example_prompt = PromptTemplate.from_template(example_template)

start_template = """Now, do this for real!

Q: {input}
A:"""
start_prompt = PromptTemplate.from_template(start_template)

input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt, pipeline_prompts=input_prompts
)

pipeline_prompt.input_variables

logger.debug(
    pipeline_prompt.format(
        person="Elon Musk",
        example_q="What's your favorite car?",
        example_a="Tesla",
        input="What's your favorite social media site?",
    )
)

"""
## Next steps

You've now learned how to compose prompts together.

Next, check out the other how-to guides on prompt templates in this section, like [adding few-shot examples to your prompt templates](/docs/how_to/few_shot_examples_chat).
"""
logger.info("## Next steps")

logger.info("\n\n[DONE]", bright=True)
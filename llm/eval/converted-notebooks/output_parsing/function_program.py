from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.anthropic import Anthropic
from jet.llm.ollama.base import Ollama
from llama_index.core.program import FunctionCallingProgram
from typing import List
from pydantic import BaseModel
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Function Calling Program for Structured Extraction
#
# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/output_parsing/function_program.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# This guide shows you how to do structured data extraction with our `FunctionCallingProgram`. Given a function-calling LLM as well as an output Pydantic class, generate a structured Pydantic object. We use three different function calling LLMs:
# - Ollama
# - Anthropic Claude
# - Mistral
#
# In terms of the target object, you can choose to directly specify `output_cls`, or specify a `PydanticOutputParser` or any other BaseOutputParser that generates a Pydantic object.
#
# in the examples below, we show you different ways of extracting into the `Album` object (which can contain a list of Song objects).
#
# **NOTE**: The `FunctionCallingProgram` only works with LLMs that natively support function calling, by inserting the schema of the Pydantic object as the "tool parameters" for a tool. For all other LLMs, please use our `LLMTextCompletionProgram`, which will directly prompt the model through text to get back a structured output.

# Define `Album` class
#
# This is a simple example of parsing an output into an `Album` schema, which can contain multiple songs.
#
# Just pass `Album` into the `output_cls` property on initialization of the `FunctionCallingProgram`.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# !pip install llama-index


# Define output schema


class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]

# Define Function Calling Program
#
# We define a function calling program with three function-calling LLMs:
# - Ollama
# - Anthropic
# - Mistral

# Function Calling Program with Ollama
#
# Here we use gpt-3.5-turbo.
#
# We demonstrate structured data extraction "single" function calling and also parallel function calling, allowing us to extract out multiple objects.

# Function Calling (Single Object)


prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)

program = FunctionCallingProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    verbose=True,
)

# Run program to get structured output.

output = program(movie_name="The Shining")

# The output is a valid Pydantic object that we can then use to call functions/APIs.

output

# Function Calling (Parallel Function Calling, Multiple Objects)

prompt_template_str = """\
Generate example albums, with an artist and a list of songs, using each movie below as inspiration. \

Here are the movies:
{movie_names}
"""
llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)

program = FunctionCallingProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    verbose=True,
    allow_parallel_tool_calls=True,
)
output = program(movie_names="The Shining, The Blair Witch Project, Saw")

output

# Function Calling Program with Anthropic
#
# Here we use Claude Sonnet (all three models support function calling).


prompt_template_str = "Generate a song about {topic}."
llm = Anthropic(model="claude-3-sonnet-20240229")

program = FunctionCallingProgram.from_defaults(
    output_cls=Song,
    prompt_template_str=prompt_template_str,
    llm=llm,
    verbose=True,
)

output = program(topic="harry potter")

output

# Function Calling Program with Mistral
#
# Here we use mistral-large.


prompt_template_str = "Generate a song about {topic}."
llm = MistralAI(model="mistral-large-latest")
program = FunctionCallingProgram.from_defaults(
    output_cls=Song,
    prompt_template_str=prompt_template_str,
    llm=llm,
    verbose=True,
)

output = program(topic="the broadway show Wicked")

output


program = LLMTextCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(output_cls=Album),
    prompt_template_str=prompt_template_str,
    verbose=True,
)

output = program(movie_name="Lord of the Rings")
output

logger.info("\n\n[DONE]", bright=True)

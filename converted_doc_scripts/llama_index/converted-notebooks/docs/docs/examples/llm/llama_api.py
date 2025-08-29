from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.openai_utils import to_openai_function
from llama_index.llms.llama_api import LlamaAPI
from llama_index.program.openai import OllamaFunctionCallingAdapterPydanticProgram
from pydantic import BaseModel
from typing import List
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/llama_api.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Llama API

[Llama API](https://www.llama-api.com/) is a hosted API for Llama 2 with function calling support.

## Setup

To start, go to https://www.llama-api.com/ to obtain an API key

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Llama API")

# %pip install llama-index-program-openai
# %pip install llama-index-llms-llama-api

# !pip install llama-index


api_key = "LL-your-key"

llm = LlamaAPI(api_key=api_key)

"""
## Basic Usage

#### Call `complete` with a prompt
"""
logger.info("## Basic Usage")

resp = llm.complete("Paul Graham is ")

logger.debug(resp)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

logger.debug(resp)

"""
## Function Calling
"""
logger.info("## Function Calling")



class Song(BaseModel):
    """A song with name and artist"""

    name: str
    artist: str


song_fn = to_openai_function(Song)

llm = LlamaAPI(api_key=api_key)
response = llm.complete("Generate a song", functions=[song_fn])
function_call = response.additional_kwargs["function_call"]
logger.debug(function_call)

"""
## Structured Data Extraction

This is a simple example of parsing an output into an `Album` schema, which can contain multiple songs.

Define output schema
"""
logger.info("## Structured Data Extraction")



class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_mins: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]

"""
Define pydantic program (llama API is OllamaFunctionCallingAdapter-compatible)
"""
logger.info("Define pydantic program (llama API is OllamaFunctionCallingAdapter-compatible)")


prompt_template_str = """\
Extract album and songs from the text provided.
For each song, make sure to specify the title and the length_mins.
{text}
"""

llm = LlamaAPI(api_key=api_key, temperature=0.0)

program = OllamaFunctionCallingAdapterPydanticProgram.from_defaults(
    output_cls=Album,
    llm=llm,
    prompt_template_str=prompt_template_str,
    verbose=True,
)

"""
Run program to get structured output.
"""
logger.info("Run program to get structured output.")

output = program(
    text="""
"Echoes of Eternity" is a compelling and thought-provoking album, skillfully crafted by the renowned artist, Seraphina Rivers. \
This captivating musical collection takes listeners on an introspective journey, delving into the depths of the human experience \
and the vastness of the universe. With her mesmerizing vocals and poignant songwriting, Seraphina Rivers infuses each track with \
raw emotion and a sense of cosmic wonder. The album features several standout songs, including the hauntingly beautiful "Stardust \
Serenade," a celestial ballad that lasts for six minutes, carrying listeners through a celestial dreamscape. "Eclipse of the Soul" \
captivates with its enchanting melodies and spans over eight minutes, inviting introspection and contemplation. Another gem, "Infinity \
Embrace," unfolds like a cosmic odyssey, lasting nearly ten minutes, drawing listeners deeper into its ethereal atmosphere. "Echoes of Eternity" \
is a masterful testament to Seraphina Rivers' artistic prowess, leaving an enduring impact on all who embark on this musical voyage through \
time and space.
"""
)

output

logger.info("\n\n[DONE]", bright=True)
from IPython.display import Markdown, display
from jet.logger import CustomLogger
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.output_parsers import BaseOutputParser
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import FunctionCallingProgram
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.prompts.default_prompts import (
DEFAULT_TEXT_QA_PROMPT_TMPL,
)
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from pydantic import BaseModel
from typing import List
import logging
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# LLM Pydantic Program - NVIDIA

This guide shows you how to generate structured data with our `LLMTextCompletionProgram`. Given an LLM as well as an output Pydantic class, generate a structured Pydantic object.

In terms of the target object, you can choose to directly specify `output_cls`, or specify a `PydanticOutputParser` or any other BaseOutputParser that generates a Pydantic object.

in the examples below, we show you different ways of extracting into the `Album` object (which can contain a list of Song objects)

## Extract into `Album` class

This is a simple example of parsing an output into an `Album` schema, which can contain multiple songs.

Just pass `Album` into the `output_cls` property on initialization of the `LLMTextCompletionProgram`.
"""
logger.info("# LLM Pydantic Program - NVIDIA")

# %pip install llama-index-readers-file llama-index-embeddings-nvidia llama-index-llms-nvidia

# import getpass

if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    logger.debug("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
#     nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith(
        "nvapi-"
    ), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key


llm = NVIDIA()

embedder = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.embed_model = embedder
Settings.llm = llm

class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]

prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
program = LLMTextCompletionProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    verbose=True,
)

"""
Run program to get structured output.
"""
logger.info("Run program to get structured output.")

output = program(movie_name="The Shining")

"""
The output is a valid Pydantic object that we can then use to call functions/APIs.
"""
logger.info("The output is a valid Pydantic object that we can then use to call functions/APIs.")

output


program = LLMTextCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(output_cls=Album),
    prompt_template_str=prompt_template_str,
    verbose=True,
)

output = program(movie_name="Lord of the Rings")
output

"""
## Define a Custom Output Parser

Sometimes you may want to parse an output your own way into a JSON object.
"""
logger.info("## Define a Custom Output Parser")



class CustomAlbumOutputParser(BaseOutputParser):
    """Custom Album output parser.

    Assume first line is name and artist.

    Assume each subsequent line is the song.

    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def parse(self, output: str) -> Album:
        """Parse output."""
        if self.verbose:
            logger.debug(f"> Raw output: {output}")
        lines = output.split("\n")
        lines = list(filter(None, (line.strip() for line in lines)))
        name, artist = lines[1].split(",")
        songs = []
        for i in range(2, len(lines)):
            title, length_seconds = lines[i].split(",")
            songs.append(Song(title=title, length_seconds=length_seconds))

        return Album(name=name, artist=artist, songs=songs)

prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\

Return answer in following format.
The first line is:
<album_name>, <album_artist>
Every subsequent line is a song with format:
<song_title>, <song_length_in_seconds>

"""
program = LLMTextCompletionProgram.from_defaults(
    output_parser=CustomAlbumOutputParser(verbose=True),
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    verbose=True,
)

output = program(movie_name="The Dark Knight")
logger.debug(output)

"""
# Function Calling Program for Structured Extraction

This guide shows you how to do structured data extraction with our `FunctionCallingProgram`. Given a function-calling LLM as well as an output Pydantic class, generate a structured Pydantic object.

in the examples below, we show you different ways of extracting into the `Album` object (which can contain a list of Song objects).

**NOTE**: The `FunctionCallingProgram` only works with LLMs that natively support function calling, by inserting the schema of the Pydantic object as the "tool parameters" for a tool. For all other LLMs, please use our `LLMTextCompletionProgram`, which will directly prompt the model through text to get back a structured output.

### Without docstring in Model
"""
logger.info("# Function Calling Program for Structured Extraction")

llm = NVIDIA(model="meta/llama-3.1-8b-instruct")

class Song(BaseModel):
    title: str
    length_seconds: int


class Album(BaseModel):
    name: str
    artist: str
    songs: List[Song]

"""
Define pydantic program
"""
logger.info("Define pydantic program")

prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""

program = FunctionCallingProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    verbose=True,
    llm=llm,
)

"""
Run program to get structured output.
"""
logger.info("Run program to get structured output.")

output = program(
    movie_name="The Shining", description="Data model for an album."
)

"""
### With docstring in Model
"""
logger.info("### With docstring in Model")

class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]

prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
program = FunctionCallingProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    verbose=True,
    llm=llm,
)

"""
Run program to get structured output.
"""
logger.info("Run program to get structured output.")

output = program(movie_name="The Shining")

"""
The output is a valid Pydantic object that we can then use to call functions/APIs.
"""
logger.info("The output is a valid Pydantic object that we can then use to call functions/APIs.")

output

"""
# Langchain Output Parsing

Download Data
"""
logger.info("# Langchain Output Parsing")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load documents, build the VectorStoreIndex
"""
logger.info("#### Load documents, build the VectorStoreIndex")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents, chunk_size=512)

"""
#### Define Query + Langchain Output Parser
"""
logger.info("#### Define Query + Langchain Output Parser")


"""
**Define custom QA and Refine Prompts**
"""

response_schemas = [
    ResponseSchema(
        name="Education",
        description=(
            "Describes the author's educational experience/background."
        ),
    ),
    ResponseSchema(
        name="Work",
        description="Describes the author's work experience/background.",
    ),
]

lc_output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)
output_parser = LangchainOutputParser(lc_output_parser)


fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
logger.debug(fmt_qa_tmpl)

"""
#### Query Index
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine(
    llm=llm,
)
response = query_engine.query(
    "What are a few things the author did growing up?",
)

logger.info("\n\n[DONE]", bright=True)
from jet.logger import CustomLogger
from llama_index.core.output_parsers import BaseOutputParser
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import LLMTextCompletionProgram
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/output_parsing/openai_pydantic_program.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# LLM Pydantic Program

This guide shows you how to generate structured data with our `LLMTextCompletionProgram`. Given an LLM as well as an output Pydantic class, generate a structured Pydantic object.

In terms of the target object, you can choose to directly specify `output_cls`, or specify a `PydanticOutputParser` or any other BaseOutputParser that generates a Pydantic object.

in the examples below, we show you different ways of extracting into the `Album` object (which can contain a list of Song objects)

## Extract into `Album` class

This is a simple example of parsing an output into an `Album` schema, which can contain multiple songs.

Just pass `Album` into the `output_cls` property on initialization of the `LLMTextCompletionProgram`.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# LLM Pydantic Program")

# !pip install llama-index



"""
Define output schema
"""
logger.info("Define output schema")

class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]

"""
Define LLM pydantic program
"""
logger.info("Define LLM pydantic program")


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

"""
### Initialize with Pydantic Output Parser

The above is equivalent to defining a Pydantic output parser and passing that in instead of the `output_cls` directly.
"""
logger.info("### Initialize with Pydantic Output Parser")


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
        name, artist = lines[0].split(",")
        songs = []
        for i in range(1, len(lines)):
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
<song_title>, <song_length_seconds>

"""
program = LLMTextCompletionProgram.from_defaults(
    output_parser=CustomAlbumOutputParser(verbose=True),
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    verbose=True,
)

output = program(movie_name="The Dark Knight")

output

logger.info("\n\n[DONE]", bright=True)
from directory import DirectoryTree, Node
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.program.openai import MLXPydanticProgram
from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import List
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/output_parsing/openai_pydantic_program.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MLX Pydantic Program

This guide shows you how to generate structured data with [new MLX API](https://openai.com/blog/function-calling-and-other-api-updates) via LlamaIndex. The user just needs to specify a Pydantic object.

We demonstrate two settings:
- Extraction into an `Album` object (which can contain a list of Song objects)
- Extraction into a `DirectoryTree` object (which can contain recursive Node objects)

## Extraction into `Album`

This is a simple example of parsing an output into an `Album` schema, which can contain multiple songs.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# MLX Pydantic Program")

# %pip install llama-index-llms-ollama
# %pip install llama-index-program-openai

# %pip install llama-index



"""
### Without docstring in Model

Define output schema (without docstring)
"""
logger.info("### Without docstring in Model")

class Song(BaseModel):
    title: str
    length_seconds: int


class Album(BaseModel):
    name: str
    artist: str
    songs: List[Song]

"""
Define openai pydantic program
"""
logger.info("Define openai pydantic program")

prompt_template_str = """\
Generate an example album, with an artist and a list of songs. \
Using the movie {movie_name} as inspiration.\
"""
program = MLXPydanticProgram.from_defaults(
    output_cls=Album, prompt_template_str=prompt_template_str, verbose=True
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
program = MLXPydanticProgram.from_defaults(
    output_cls=Album, prompt_template_str=prompt_template_str, verbose=True
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
## Stream partial intermediate Pydantic Objects

Instead of waiting for the Function Call to generate the entire JSON, we can use the `stream_partial_objects()` method of the `program` to stream valid intermediate instances of the Pydantic Output class as soon as they're available 🔥

First let's define the Output Pydantic class
"""
logger.info("## Stream partial intermediate Pydantic Objects")



class CharacterInfo(BaseModel):
    """Information about a character."""

    character_name: str
    name: str = Field(..., description="Name of the actor/actress")
    hometown: str


class Characters(BaseModel):
    """List of characters."""

    characters: list[CharacterInfo] = Field(default_factory=list)

"""
Now we'll initialilze the program with prompt template
"""
logger.info("Now we'll initialilze the program with prompt template")


prompt_template_str = "Information about 3 characters from the movie: {movie}"

program = MLXPydanticProgram.from_defaults(
    output_cls=Characters, prompt_template_str=prompt_template_str
)

"""
Finally we stream the partial objects using the `stream_partial_objects()` method
"""
logger.info("Finally we stream the partial objects using the `stream_partial_objects()` method")

for partial_object in program.stream_partial_objects(movie="Harry Potter"):
    logger.debug(partial_object)

"""
## Extracting List of `Album` (with Parallel Function Calling)

With the latest [parallel function calling](https://platform.openai.com/docs/guides/function-calling/parallel-function-calling) feature from MLX, we can simultaneously extract multiple structured data from a single prompt!

To do this, we need to:
1. pick one of the latest models (e.g. `gpt-3.5-turbo-1106`), and 
2. set `allow_multiple` to True in our `MLXPydanticProgram` (if not, it will only return the first object, and raise a warning).
"""
logger.info("## Extracting List of `Album` (with Parallel Function Calling)")


prompt_template_str = """\
Generate 4 albums about spring, summer, fall, and winter.
"""
program = MLXPydanticProgram.from_defaults(
    output_cls=Album,
    llm=MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
    prompt_template_str=prompt_template_str,
    allow_multiple=True,
    verbose=True,
)

output = program()

"""
The output is a list of valid Pydantic object.
"""
logger.info("The output is a list of valid Pydantic object.")

output

"""
## Extraction into `Album` (Streaming)

We also support streaming a list of objects through our `stream_list` function.

Full credits to this idea go to `openai_function_call` repo: https://github.com/jxnl/openai_function_call/tree/main/examples/streaming_multitask
"""
logger.info("## Extraction into `Album` (Streaming)")

prompt_template_str = "{input_str}"
program = MLXPydanticProgram.from_defaults(
    output_cls=Album,
    prompt_template_str=prompt_template_str,
    verbose=False,
)

output = program.stream_list(
    input_str="make up 5 random albums",
)
for obj in output:
    logger.debug(obj.json(indent=2))

"""
## Extraction into `DirectoryTree` object

This is directly inspired by jxnl's awesome repo here: https://github.com/jxnl/openai_function_call.

That repository shows how you can use MLX's function API to parse recursive Pydantic objects. The main requirement is that you want to "wrap" a recursive Pydantic object with a non-recursive one.

Here we show an example in a "directory" setting, where a `DirectoryTree` object wraps recursive `Node` objects, to parse a file structure.
"""
logger.info("## Extraction into `DirectoryTree` object")


DirectoryTree.schema()

program = MLXPydanticProgram.from_defaults(
    output_cls=DirectoryTree,
    prompt_template_str="{input_str}",
    verbose=True,
)

input_str = """
root
├── folder1
│   ├── file1.txt
│   └── file2.txt
└── folder2
    ├── file3.txt
    └── subfolder1
        └── file4.txt
"""

output = program(input_str=input_str)

"""
The output is a full DirectoryTree structure with recursive `Node` objects.
"""
logger.info("The output is a full DirectoryTree structure with recursive `Node` objects.")

output

logger.info("\n\n[DONE]", bright=True)
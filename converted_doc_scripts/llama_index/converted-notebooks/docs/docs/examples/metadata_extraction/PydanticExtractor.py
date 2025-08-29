from jet.logger import CustomLogger
from llama_index.core.extractors import PydanticProgramExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.program.openai import OllamaFunctionCallingAdapterPydanticProgram
from llama_index.readers.web import SimpleWebPageReader
from pydantic import BaseModel, Field
from typing import List
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/metadata_extraction/PydanticExtractor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Pydantic Extractor

Here we test out the capabilities of our `PydanticProgramExtractor` - being able to extract out an entire Pydantic object using an LLM (either a standard text completion LLM or a function calling LLM).

The advantage of this over using a "single" metadata extractor is that we can extract multiple entities with a single LLM call.

## Setup
"""
logger.info("# Pydantic Extractor")

# %pip install llama-index-readers-web
# %pip install llama-index-program-openai

# import nest_asyncio

# nest_asyncio.apply()


# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
# openai.api_key = os.getenv("OPENAI_API_KEY")

"""
### Setup the Pydantic Model

Here we define a basic structured schema that we want to extract. It contains:

- entities: unique entities in a text chunk
- summary: a concise summary of the text chunk
- contains_number: whether the chunk contains numbers

This is obviously a toy schema. We'd encourage you to be creative about the type of metadata you'd want to extract!
"""
logger.info("### Setup the Pydantic Model")


class NodeMetadata(BaseModel):
    """Node metadata."""

    entities: List[str] = Field(
        ..., description="Unique entities in this text chunk."
    )
    summary: str = Field(
        ..., description="A concise summary of this text chunk."
    )
    contains_number: bool = Field(
        ...,
        description=(
            "Whether the text chunk contains any numbers (ints, floats, etc.)"
        ),
    )

"""
### Setup the Extractor

Here we setup the metadata extractor. Note that we provide the prompt template for visibility into what's going on.
"""
logger.info("### Setup the Extractor")


EXTRACT_TEMPLATE_STR = """\
Here is the content of the section:
----------------
{context_str}
----------------
Given the contextual information, extract out a {class_name} object.\
"""

openai_program = OllamaFunctionCallingAdapterPydanticProgram.from_defaults(
    output_cls=NodeMetadata,
    prompt_template_str="{input}",
)

program_extractor = PydanticProgramExtractor(
    program=openai_program, input_key="input", show_progress=True
)

"""
### Load in Data

We load in Eugene's essay (https://eugeneyan.com/writing/llm-patterns/) using our LlamaHub SimpleWebPageReader.
"""
logger.info("### Load in Data")


reader = SimpleWebPageReader(html_to_text=True)
docs = reader.load_data(urls=["https://eugeneyan.com/writing/llm-patterns/"])


node_parser = SentenceSplitter(chunk_size=1024)

pipeline = IngestionPipeline(transformations=[node_parser, program_extractor])

orig_nodes = pipeline.run(documents=docs)

orig_nodes

"""
## Extract Metadata

Now that we've setup the metadata extractor and the data, we're ready to extract some metadata! 

We see that the pydantic feature extractor is able to extract *all* metadata from a given chunk in a single LLM call.
"""
logger.info("## Extract Metadata")

sample_entry = program_extractor.extract(orig_nodes[0:1])[0]

display(sample_entry)

new_nodes = program_extractor.process_nodes(orig_nodes)

display(new_nodes[5:7])

logger.info("\n\n[DONE]", bright=True)
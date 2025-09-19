from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.extractors.marvin import MarvinMetadataExtractor
from pprint import pprint
from pydantic import BaseModel, Field
import marvin
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
# Metadata Extraction and Augmentation w/ Marvin

This notebook walks through using [`Marvin`](https://github.com/PrefectHQ/marvin) to extract and augment metadata from text. Marvin uses the LLM to identify and extract metadata.  Metadata can be anything from additional and enhanced questions and answers to business object identification and elaboration.  This notebook will demonstrate pulling out and elaborating on Sports Supplement information in a csv document.

Note: You will need to supply a valid open ai key below to run this notebook.

## Setup
"""
logger.info("# Metadata Extraction and Augmentation w/ Marvin")

# %pip install llama-index-llms-ollama
# %pip install llama-index-extractors-marvin




# import nest_asyncio

# nest_asyncio.apply()


# os.environ["OPENAI_API_KEY"] = "sk-..."

documents = SimpleDirectoryReader("data").load_data()

documents[0].text = documents[0].text[:10000]



# marvin.settings.openai.api_key = os.environ["OPENAI_API_KEY"]
marvin.settings.openai.chat.completions.model = "gpt-4o"


class SportsSupplement(BaseModel):
    name: str = Field(..., description="The name of the sports supplement")
    description: str = Field(
        ..., description="A description of the sports supplement"
    )
    pros_cons: str = Field(
        ..., description="The pros and cons of the sports supplement"
    )

node_parser = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)

metadata_extractor = MarvinMetadataExtractor(
    marvin_model=SportsSupplement
)  # let's extract custom entities for each node.


pipeline = IngestionPipeline(transformations=[node_parser, metadata_extractor])

nodes = pipeline.run(documents=documents, show_progress=True)


for i in range(5):
    plogger.debug(nodes[i].metadata)

logger.info("\n\n[DONE]", bright=True)
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.program.lmformatenforcer import (
LMFormatEnforcerPydanticProgram,
)
from pydantic import BaseModel, Field
from typing import List
import os
import shutil
import sys


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/output_parsing/lmformatenforcer_pydantic_program.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# LM Format Enforcer Pydantic Program

Generate structured data with [**lm-format-enforcer**](https://github.com/noamgat/lm-format-enforcer) via LlamaIndex.  


With lm-format-enforcer, you can guarantee the output structure is correct by *forcing* the LLM to output desired tokens.  
This is especialy helpful when you are using lower-capacity model (e.g. the current open source models), which otherwise would struggle to generate valid output that fits the desired output schema.

[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) supports regular expressions and JSON Schema, this demo focuses on JSON Schema. For regular expressions, see the [sample regular expressions notebook](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/output_parsing/lmformatenforcer_regular_expressions.ipynb).

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# LM Format Enforcer Pydantic Program")

# %pip install llama-index-program-lmformatenforcer
# %pip install llama-index-llms-llama-cpp

# !pip install llama-index lm-format-enforcer llama-cpp-python




"""
Define output schema
"""
logger.info("Define output schema")

class Song(BaseModel):
    title: str
    length_seconds: int


class Album(BaseModel):
    name: str
    artist: str
    songs: List[Song] = Field(min_items=3, max_items=10)

"""
Create the program. We use `LlamaCPP` as the LLM in this demo, but `HuggingFaceLLM` is also supported.

Note that the prompt template has two parameters:
- `movie_name` which will be used in the function called
- `json_schema` which will automatically have the JSON Schema of the output class injected into it.
"""
logger.info("Create the program. We use `LlamaCPP` as the LLM in this demo, but `HuggingFaceLLM` is also supported.")


llm = LlamaCPP()

program = LMFormatEnforcerPydanticProgram(
    output_cls=Album,
    prompt_template_str=(
        "Your response should be according to the following json schema: \n"
        "{json_schema}\n"
        "Generate an example album, with an artist and a list of songs. Using"
        " the movie {movie_name} as inspiration. "
    ),
    llm=llm,
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

logger.info("\n\n[DONE]", bright=True)
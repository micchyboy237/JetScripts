from jet.logger import logger
from pydantic import BaseModel
from typing import List
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
# (Deprecated) Query Engines + Pydantic Outputs

<Aside type="tip">
This guide references a deprecated method of extracting structured outputs in a RAG workflow. Check out our [structured output starter guide](/python/examples/structured_outputs/structured_outputs) for more details.
</Aside>

Using `index.as_query_engine()` and it's underlying `RetrieverQueryEngine`, we can support structured pydantic outputs without an additional LLM calls (in contrast to a typical output parser.)

Every query engine has support for integrated structured responses using the following `response_mode`s in `RetrieverQueryEngine`:

- `refine`
- `compact`
- `tree_summarize`
- `accumulate` (beta, requires extra parsing to convert to objects)
- `compact_accumulate` (beta, requires extra parsing to convert to objects)

Under the hood, this uses `OpenAIPydanitcProgam` or `LLMTextCompletionProgram` depending on which LLM you've setup. If there are intermediate LLM responses (i.e. during `refine` or `tree_summarize` with multiple LLM calls), the pydantic object is injected into the next LLM prompt as a JSON object.

## Usage Pattern

First, you need to define the object you want to extract.
"""
logger.info("# (Deprecated) Query Engines + Pydantic Outputs")



class Biography(BaseModel):
    """Data model for a biography."""

    name: str
    best_known_for: List[str]
    extra_info: str

"""
Then, you create your query engine.
"""
logger.info("Then, you create your query engine.")

query_engine = index.as_query_engine(
    response_mode="tree_summarize", output_cls=Biography
)

"""
Lastly, you can get a response and inspect the output.
"""
logger.info("Lastly, you can get a response and inspect the output.")

response = query_engine.query("Who is Paul Graham?")

logger.debug(response.name)
logger.debug(response.best_known_for)
logger.debug(response.extra_info)

"""
## Modules

Detailed usage is available in the notebooks below:

- [Structured Outputs with a Query Engine](/python/examples/query_engine/pydantic_query_engine)
- [Structured Outputs with a Tree Summarize](/python/examples/response_synthesizers/pydantic_tree_summarize)
"""
logger.info("## Modules")

logger.info("\n\n[DONE]", bright=True)
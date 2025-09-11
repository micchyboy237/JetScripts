from jet.logger import logger
from langchain_community.embeddings import LlamafileEmbeddings
from langchain_community.llms.llamafile import Llamafile
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
# llamafile

>[llamafile](https://github.com/Mozilla-Ocho/llamafile) lets you distribute and run LLMs
> with a single file.

>`llamafile` makes open LLMs much more accessible to both developers and end users.
> `llamafile` is doing that by combining [llama.cpp](https://github.com/ggerganov/llama.cpp) with
> [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one framework that collapses
> all the complexity of LLMs down to a single-file executable (called a "llamafile")
> that runs locally on most computers, with no installation.


## Installation and Setup

See the [installation instructions](https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#quickstart).

## LLMs

See a [usage example](/docs/integrations/llms/llamafile).
"""
logger.info("# llamafile")


"""
## Embedding models

See a [usage example](/docs/integrations/text_embedding/llamafile).
"""
logger.info("## Embedding models")


logger.info("\n\n[DONE]", bright=True)
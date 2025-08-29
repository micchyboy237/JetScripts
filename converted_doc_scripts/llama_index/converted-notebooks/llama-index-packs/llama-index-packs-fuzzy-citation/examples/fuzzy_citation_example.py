from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex
from llama_index.core.llama_pack import download_llama_pack
from llama_index.readers.file import UnstructuredReader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Fuzzy Citation Query Engine

This notebook walks through using the `FuzzyCitationEnginePack`, which can wrap any existing query engine and post-process the response object to include direct sentence citations, identified using fuzzy-matching.

## Setup
"""
logger.info("# Fuzzy Citation Query Engine")

# %pip install llama-index-readers-file


# os.environ["OPENAI_API_KEY"] = "sk-..."

# !mkdir -p 'data/'
# !curl 'https://arxiv.org/pdf/2307.09288.pdf' -o 'data/llama2.pdf'

# !pip install unstructured[pdf]



documents = UnstructuredReader().load_data("data/llama2.pdf")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

"""
## Run the FuzzyCitationEnginePack

The `FuzzyCitationEnginePack` can wrap any existing query engine.
"""
logger.info("## Run the FuzzyCitationEnginePack")


FuzzyCitationEnginePack = download_llama_pack("FuzzyCitationEnginePack", "./fuzzy_pack")

fuzzy_engine_pack = FuzzyCitationEnginePack(query_engine, threshold=50)

response = fuzzy_engine_pack.run("How was Llama2 pretrained?")

logger.debug(str(response))

"""
### Compare response to citation sentences
"""
logger.info("### Compare response to citation sentences")

for response_sentence, node_chunk in response.metadata.keys():
    logger.debug("Response Sentence:\n", response_sentence)
    logger.debug("\nRelevant Node Chunk:\n", node_chunk)
    logger.debug("----------------")

"""
So if we compare the original LLM output:

```
Llama 2 was pretrained using an optimized auto-regressive transformer. The pretraining approach involved robust data cleaning, updating the data mixes, training on 40% more total tokens, doubling the context length, and using grouped-query attention (GQA) to improve inference scalability for larger models. The training corpus included a new mix of data from publicly available sources, excluding data from Meta's products or services. The pretraining methodology and training details are described in more detail in the provided context.
```

With the generated fuzzy matches above, we can clearly see where each sentence came from!

### [Advanced] Inspect citation metadata

Using the citation metadata, we can get the exact character location of the response from the original document!
"""
logger.info("### [Advanced] Inspect citation metadata")

for chunk_info in response.metadata.values():
    start_char_idx = chunk_info["start_char_idx"]
    end_char_idx = chunk_info["end_char_idx"]

    node = chunk_info["node"]
    node_start_char_idx = node.start_char_idx
    node_end_char_idx = node.end_char_idx

    document_start_char_idx = start_char_idx + node_start_char_idx
    document_end_char_idx = document_start_char_idx + (end_char_idx - start_char_idx)
    text = documents[0].text[document_start_char_idx:document_end_char_idx]

    logger.debug(text)
    logger.debug(node.metadata)
    logger.debug("----------------")

"""
## Try a random question

If we ask a question unrelated to the data in the index, we should not have any matching citaitons (in most cases).
"""
logger.info("## Try a random question")

response = fuzzy_engine_pack.run("Where is San Francisco located?")

logger.debug(len(response.metadata.keys()))

logger.info("\n\n[DONE]", bright=True)
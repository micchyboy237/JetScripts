from jet.adapters.langchain.chat_ollama.embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_experimental.text_splitter import SemanticChunker
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
# How to split text based on semantic similarity

Taken from Greg Kamradt's wonderful notebook:
[5_Levels_Of_Text_Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)

All credit to him.

This guide covers how to split chunks based on their semantic similarity. If embeddings are sufficiently far apart, chunks are split.

At a high level, this splits into sentences, then groups into groups of 3
sentences, and then merges one that are similar in the embedding space.

## Install Dependencies
"""
logger.info("# How to split text based on semantic similarity")

# !pip install --quiet langchain_experimental jet.adapters.langchain.chat_ollama

"""
## Load Example Data
"""
logger.info("## Load Example Data")

with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

"""
## Create Text Splitter

To instantiate a [SemanticChunker](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html), we must specify an embedding model. Below we will use [OllamaEmbeddings](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.ollama.OllamaEmbeddings.html).
"""
logger.info("## Create Text Splitter")


text_splitter = SemanticChunker(OllamaEmbeddings(model="mxbai-embed-large"))

"""
## Split Text

We split text in the usual way, e.g., by invoking `.create_documents` to create LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects:
"""
logger.info("## Split Text")

docs = text_splitter.create_documents([state_of_the_union])
logger.debug(docs[0].page_content)

"""
## Breakpoints

This chunker works by determining when to "break" apart sentences. This is done by looking for differences in embeddings between any two sentences. When that difference is past some threshold, then they are split.

There are a few ways to determine what that threshold is, which are controlled by the `breakpoint_threshold_type` kwarg.

Note: if the resulting chunk sizes are too small/big, the additional kwargs `breakpoint_threshold_amount` and `min_chunk_size` can be used for adjustments.

### Percentile

The default way to split is based on percentile. In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split. The default value for X is 95.0 and can be adjusted by the keyword argument `breakpoint_threshold_amount` which expects a number between 0.0 and 100.0.
"""
logger.info("## Breakpoints")

text_splitter = SemanticChunker(
    OllamaEmbeddings(model="mxbai-embed-large"), breakpoint_threshold_type="percentile"
)

docs = text_splitter.create_documents([state_of_the_union])
logger.debug(docs[0].page_content)

logger.debug(len(docs))

"""
### Standard Deviation

In this method, any difference greater than X standard deviations is split. The default value for X is 3.0 and can be adjusted by the keyword argument `breakpoint_threshold_amount`.
"""
logger.info("### Standard Deviation")

text_splitter = SemanticChunker(
    OllamaEmbeddings(model="mxbai-embed-large"), breakpoint_threshold_type="standard_deviation"
)

docs = text_splitter.create_documents([state_of_the_union])
logger.debug(docs[0].page_content)

logger.debug(len(docs))

"""
### Interquartile

In this method, the interquartile distance is used to split chunks. The interquartile range can be scaled by the keyword argument `breakpoint_threshold_amount`, the default value is 1.5.
"""
logger.info("### Interquartile")

text_splitter = SemanticChunker(
    OllamaEmbeddings(model="mxbai-embed-large"), breakpoint_threshold_type="interquartile"
)

docs = text_splitter.create_documents([state_of_the_union])
logger.debug(docs[0].page_content)

logger.debug(len(docs))

"""
### Gradient

In this method, the gradient of distance is used to split chunks along with the percentile method. This method is useful when chunks are highly correlated with each other or specific to a domain e.g. legal or medical. The idea is to apply anomaly detection on gradient array so that the distribution become wider and easy to identify boundaries in highly semantic data.
Similar to the percentile method, the split can be adjusted by the keyword argument `breakpoint_threshold_amount` which expects a number between 0.0 and 100.0, the default value is 95.0.
"""
logger.info("### Gradient")

text_splitter = SemanticChunker(
    OllamaEmbeddings(model="mxbai-embed-large"), breakpoint_threshold_type="gradient"
)

docs = text_splitter.create_documents([state_of_the_union])
logger.debug(docs[0].page_content)

logger.debug(len(docs))

logger.info("\n\n[DONE]", bright=True)
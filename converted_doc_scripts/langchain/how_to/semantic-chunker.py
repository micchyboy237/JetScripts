from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from langchain_experimental.text_splitter import SemanticChunker
from jet.llm.ollama.base_langchain import OllamaEmbeddings

initialize_ollama_settings()

"""
# How to split text based on semantic similarity

Taken from Greg Kamradt's wonderful notebook:
[5_Levels_Of_Text_Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)

All credit to him.

This guide covers how to split chunks based on their semantic similarity. If embeddings are sufficiently far apart, chunks are split.

At a high level, this splits into sentences, then groups into groups of 3
sentences, and then merges one that are similar in the embedding space.
"""

"""
## Install Dependencies
"""

# !pip install --quiet langchain_experimental jet.llm.ollama.base_langchain

"""
## Load Example Data
"""

data_file = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/langchain/docs/docs/how_to/state_of_the_union.txt"

with open(data_file) as f:
    state_of_the_union = f.read()

"""
## Create Text Splitter
"""

"""
To instantiate a [SemanticChunker](https://python.langchain.com/api_reference/experimental/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html), we must specify an embedding model. Below we will use [OllamaEmbeddings](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.openai.OllamaEmbeddings.html).
"""


text_splitter = SemanticChunker(OllamaEmbeddings(model="nomic-embed-text"))

"""
## Split Text

We split text in the usual way, e.g., by invoking `.create_documents` to create LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects:
"""

logger.newline()
logger.info("Guide: Split Text")
docs = text_splitter.create_documents([state_of_the_union])
logger.success(docs[0].page_content)

logger.debug(len(docs))

"""
## Breakpoints

This chunker works by determining when to "break" apart sentences. This is done by looking for differences in embeddings between any two sentences. When that difference is past some threshold, then they are split.

There are a few ways to determine what that threshold is, which are controlled by the `breakpoint_threshold_type` kwarg.

Note: if the resulting chunk sizes are too small/big, the additional kwargs `breakpoint_threshold_amount` and `min_chunk_size` can be used for adjustments.

### Percentile

The default way to split is based on percentile. In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split. The default value for X is 95.0 and can be adjusted by the keyword argument `breakpoint_threshold_amount` which expects a number between 0.0 and 100.0.
"""

logger.newline()
logger.info("Guide: Percentile")
text_splitter = SemanticChunker(
    OllamaEmbeddings(model="nomic-embed-text"), breakpoint_threshold_type="percentile"
)

docs = text_splitter.create_documents([state_of_the_union])
logger.success(docs[0].page_content)

logger.debug(len(docs))

"""
### Standard Deviation

In this method, any difference greater than X standard deviations is split. The default value for X is 3.0 and can be adjusted by the keyword argument `breakpoint_threshold_amount`.
"""

logger.newline()
logger.info("Guide: Standard Deviation")
text_splitter = SemanticChunker(
    OllamaEmbeddings(model="nomic-embed-text"), breakpoint_threshold_type="standard_deviation"
)

docs = text_splitter.create_documents([state_of_the_union])
logger.success(docs[0].page_content)

logger.debug(len(docs))

"""
### Interquartile

In this method, the interquartile distance is used to split chunks. The interquartile range can be scaled by the keyword argument `breakpoint_threshold_amount`, the default value is 1.5.
"""

logger.newline()
logger.info("Guide: Interquartile")
text_splitter = SemanticChunker(
    OllamaEmbeddings(model="nomic-embed-text"), breakpoint_threshold_type="interquartile"
)

docs = text_splitter.create_documents([state_of_the_union])
logger.success(docs[0].page_content)

logger.debug(len(docs))

"""
### Gradient

In this method, the gradient of distance is used to split chunks along with the percentile method. This method is useful when chunks are highly correlated with each other or specific to a domain e.g. legal or medical. The idea is to apply anomaly detection on gradient array so that the distribution become wider and easy to identify boundaries in highly semantic data.
Similar to the percentile method, the split can be adjusted by the keyword argument `breakpoint_threshold_amount` which expects a number between 0.0 and 100.0, the default value is 95.0.
"""

logger.newline()
logger.info("Guide: Gradient")
text_splitter = SemanticChunker(
    OllamaEmbeddings(model="nomic-embed-text"), breakpoint_threshold_type="gradient"
)

docs = text_splitter.create_documents([state_of_the_union])
logger.success(docs[0].page_content)

logger.debug(len(docs))

logger.info("\n\n[DONE]", bright=True)

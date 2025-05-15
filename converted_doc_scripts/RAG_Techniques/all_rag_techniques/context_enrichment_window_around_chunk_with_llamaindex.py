from dotenv import load_dotenv
from jet.llm.ollama.base import Ollama
from jet.llm.ollama.base import OllamaEmbedding
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from pprint import pprint
import faiss
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join(script_dir, "generated", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/context_enrichment_window_around_chunk_with_llamaindex.ipynb)

# Context Enrichment Window for Document Retrieval

## Overview

This code implements a context enrichment window technique for document retrieval in a vector database. It enhances the standard retrieval process by adding surrounding context to each retrieved chunk, improving the coherence and completeness of the returned information.

## Motivation

Traditional vector search often returns isolated chunks of text, which may lack necessary context for full understanding. This approach aims to provide a more comprehensive view of the retrieved information by including neighboring text chunks.

## Key Components

1. PDF processing and text chunking
2. Vector store creation using FAISS and Ollama embeddings
3. Custom retrieval function with context window
4. Comparison between standard and context-enriched retrieval

## Method Details

### Document Preprocessing

1. The PDF is read and converted to a string.
2. The text is split into chunks with surrounding sentences

### Vector Store Creation

1. Ollama embeddings are used to create vector representations of the chunks.
2. A FAISS vector store is created from these embeddings.

### Context-Enriched Retrieval

LlamaIndex has a special parser for such task. [SentenceWindowNodeParser](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#sentencewindownodeparser) this parser splits documents into sentences. But the resulting nodes inculde the surronding senteces with a relation structure. Then, on the query [MetadataReplacementPostProcessor](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/#metadatareplacementpostprocessor) helps connecting back these related sentences.

### Retrieval Comparison

The notebook includes a section to compare standard retrieval with the context-enriched approach.

## Benefits of this Approach

1. Provides more coherent and contextually rich results
2. Maintains the advantages of vector search while mitigating its tendency to return isolated text fragments
3. Allows for flexible adjustment of the context window size

## Conclusion

This context enrichment window technique offers a promising way to improve the quality of retrieved information in vector-based document search systems. By providing surrounding context, it helps maintain the coherence and completeness of the retrieved information, potentially leading to better understanding and more accurate responses in downstream tasks such as question answering.

<div style="text-align: center;">

<img src="../images/vector-search-comparison_context_enrichment.svg" alt="context enrichment window" style="width:70%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Context Enrichment Window for Document Retrieval")

# !pip install faiss-cpu llama-index python-dotenv


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

EMBED_DIMENSION = 512
Settings.llm = Ollama(
    model="llama3.2", request_timeout=300.0, context_window=4096)
Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

"""
### Read docs
"""
logger.info("### Read docs")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/"
reader = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = reader.load_data()
logger.debug(documents[0])

"""
### Create vector store and retriever
"""
logger.info("### Create vector store and retriever")

fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=fais_index)

"""
## Ingestion Pipelines

### Ingestion Pipeline with Sentence Splitter
"""
logger.info("## Ingestion Pipelines")

base_pipeline = IngestionPipeline(
    transformations=[SentenceSplitter()],
    vector_store=vector_store
)

base_nodes = base_pipeline.run(documents=documents)

"""
### Ingestion Pipeline with Sentence Window
"""
logger.info("### Ingestion Pipeline with Sentence Window")

node_parser = SentenceWindowNodeParser(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_sentence"
)

pipeline = IngestionPipeline(
    transformations=[node_parser],
    vector_store=vector_store,
)

windowed_nodes = pipeline.run(documents=documents)

"""
## Querying
"""
logger.info("## Querying")

query = "Explain the role of deforestation and fossil fuels in climate change"

"""
### Querying *without* Metadata Replacement
"""
logger.info("### Querying *without* Metadata Replacement")

base_index = VectorStoreIndex(base_nodes)

base_query_engine = base_index.as_query_engine(
    similarity_top_k=1,
)

base_response = base_query_engine.query(query)

logger.debug(base_response)

"""
#### Print Metadata of the Retrieved Node
"""
logger.info("#### Print Metadata of the Retrieved Node")

plogger.debug(base_response.source_nodes[0].node.metadata)

"""
### Querying with Metadata Replacement
"Metadata replacement" intutively might sound a little off topic since we're working on the base sentences. But LlamaIndex stores these "before/after sentences" in the metadata data of the nodes. Therefore to build back up these windows of sentences we need Metadata replacement post processor.
"""
logger.info("### Querying with Metadata Replacement")

windowed_index = VectorStoreIndex(windowed_nodes)

windowed_query_engine = windowed_index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[
        MetadataReplacementPostProcessor(
            # `window_metadata_key` key defined in SentenceWindowNodeParser
            target_metadata_key="window"
        )
    ],
)

windowed_response = windowed_query_engine.query(query)

logger.debug(windowed_response)

"""
#### Print Metadata of the Retrieved Node
"""
logger.info("#### Print Metadata of the Retrieved Node")

plogger.debug(windowed_response.source_nodes[0].node.metadata)

logger.info("\n\n[DONE]", bright=True)

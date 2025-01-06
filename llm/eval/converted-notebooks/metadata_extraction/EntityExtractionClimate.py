from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
import random
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.extractors.entity import EntityExtractor
import os
from span_marker.tokenizer import SpanMarkerTokenizer, SpanMarkerConfig
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/metadata_extraction/EntityExtractionClimate.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Entity Metadata Extraction
#
# In this demo, we use the new `EntityExtractor` to extract entities from each node, stored in metadata. The default model is `tomaarsen/span-marker-mbert-base-multinerd`, which is downloaded an run locally from [HuggingFace](https://huggingface.co/tomaarsen/span-marker-mbert-base-multinerd).
#
# For more information on metadata extraction in LlamaIndex, see our [documentation](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_metadata_extractor.html).

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %pip install llama-index-llms-ollama
# %pip install llama-index-extractors-entity

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Setup the Extractor and Parser

ENTITY_MODEL = "tomaarsen/span-marker-mbert-base-multinerd"

entity_extractor = EntityExtractor(
    model_name=ENTITY_MODEL,
    prediction_threshold=0.5,
    # include the entity label in the metadata (can be erroneous)
    label_entities=False,
    device="cpu",  # set to "cuda" if you have a GPU
    span_joiner="",
)

node_parser = SentenceSplitter()

transformations = [node_parser, entity_extractor]

# Load the data
#
# Here, we will download the 2023 IPPC Climate Report - Chapter 3 on Oceans and Coastal Ecosystems (172 Pages)

# !curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf

# Next, load the documents.


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/summaries").load_data()
print("Documents:", len(documents))
# Extracting Metadata
#
# Now, this is a pretty long document. Since we are not running on CPU, for now, we will only run on a subset of documents. Feel free to run it on all documents on your own though!


# random.seed(42)
# documents = random.sample(documents, len(documents))

pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(documents=documents, show_progress=True)

# Examine the outputs
print("Nodes:", len(nodes))
# View first item
print(nodes[0])

# Try a Query!
Settings.llm = Ollama(model="llama3.1", request_timeout=300.0,
                      context_window=4096, temperature=0.2)

index = VectorStoreIndex(nodes=nodes)
query_engine = index.as_query_engine()
response = query_engine.query("Tell me about yourself.")
print(response)

# Contrast without metadata
#
# Here, we re-construct the index, but without metadata
for node in nodes:
    node.metadata.pop("entities", None)

index = VectorStoreIndex(nodes=nodes)
query_engine = index.as_query_engine()
response = query_engine.query("Tell me about yourself.")
print(response)

# As we can see, our metadata-enriched index is able to fetch more relevant information.

logger.info("\n\n[DONE]", bright=True)

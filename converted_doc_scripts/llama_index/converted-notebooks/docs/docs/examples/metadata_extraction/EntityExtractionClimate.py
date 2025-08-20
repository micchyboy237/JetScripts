from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.extractors.entity import EntityExtractor
import os
import random
import shutil


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/metadata_extraction/EntityExtractionClimate.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Entity Metadata Extraction

In this demo, we use the new `EntityExtractor` to extract entities from each node, stored in metadata. The default model is `tomaarsen/span-marker-mbert-base-multinerd`, which is downloaded an run locally from [HuggingFace](https://huggingface.co/tomaarsen/span-marker-mbert-base-multinerd).

For more information on metadata extraction in LlamaIndex, see our [documentation](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_metadata_extractor.html).

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Entity Metadata Extraction")

# %pip install llama-index-llms-ollama
# %pip install llama-index-extractors-entity

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

"""
## Setup the Extractor and Parser
"""
logger.info("## Setup the Extractor and Parser")


entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_entities=False,  # include the entity label in the metadata (can be erroneous)
    device="cpu",  # set to "cuda" if you have a GPU
)

node_parser = SentenceSplitter()

transformations = [node_parser, entity_extractor]

"""
## Load the data

Here, we will download the 2023 IPPC Climate Report - Chapter 3 on Oceans and Coastal Ecosystems (172 Pages)
"""
logger.info("## Load the data")

# !curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf

"""
Next, load the documents.
"""
logger.info("Next, load the documents.")


documents = SimpleDirectoryReader(
    input_files=["./IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

"""
## Extracting Metadata

Now, this is a pretty long document. Since we are not running on CPU, for now, we will only run on a subset of documents. Feel free to run it on all documents on your own though!
"""
logger.info("## Extracting Metadata")



random.seed(42)
documents = random.sample(documents, 100)

pipeline = IngestionPipeline(transformations=transformations)

nodes = pipeline.run(documents=documents)

"""
### Examine the outputs
"""
logger.info("### Examine the outputs")

samples = random.sample(nodes, 5)
for node in samples:
    logger.debug(node.metadata)

"""
## Try a Query!
"""
logger.info("## Try a Query!")


Settings.llm = MLXLlamaIndexLLMAdapter(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.2)

index = VectorStoreIndex(nodes=nodes)

query_engine = index.as_query_engine()
response = query_engine.query("What is said by Fox-Kemper?")
logger.debug(response)

"""
### Contrast without metadata

Here, we re-construct the index, but without metadata
"""
logger.info("### Contrast without metadata")

for node in nodes:
    node.metadata.pop("entities", None)

logger.debug(nodes[0].metadata)

index = VectorStoreIndex(nodes=nodes)

query_engine = index.as_query_engine()
response = query_engine.query("What is said by Fox-Kemper?")
logger.debug(response)

"""
As we can see, our metadata-enriched index is able to fetch more relevant information.
"""
logger.info("As we can see, our metadata-enriched index is able to fetch more relevant information.")

logger.info("\n\n[DONE]", bright=True)
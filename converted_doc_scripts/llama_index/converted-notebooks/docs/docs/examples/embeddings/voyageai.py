from IPython.display import Markdown, display
from PIL import Image
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.embeddings.voyageai import VoyageEmbedding
import logging
import matplotlib.pyplot as plt
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/voyageai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# VoyageAI Embeddings

New VoyageAI Embedding models natively supports float, int8, binary and ubinary embeddings. Please check `output_dtype` description [here](https://docs.voyageai.com/docs/embeddings) for more details.

In this notebook, we will demonstrate using `VoyageAI Embeddings` with different `models`, `input_types` and `embedding_types`.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# VoyageAI Embeddings")

# %pip install llama-index-llms-ollama
# %pip install llama-index-embeddings-voyageai

# !pip install llama-index

"""
#### With latest `voyage-3` embeddings.


The default `embedding_type` is `float`.
"""
logger.info("#### With latest `voyage-3` embeddings.")


embed_model = VoyageEmbedding(
    voyage_api_key="<YOUR_VOYAGE_API_KEY>",
    model_name="voyage-3",
)

embeddings = embed_model.get_text_embedding("Hello VoyageAI!")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
##### Let's check With `int8` embedding_type with `voyage-3-large` model
"""
logger.info(
    "##### Let's check With `int8` embedding_type with `voyage-3-large` model")

embed_model = VoyageEmbedding(
    voyage_api_key="<YOUR_VOYAGE_API_KEY>",
    model_name="voyage-3-large",
    output_dtype="int8",
    truncation=False,
)

embeddings = embed_model.get_text_embedding("Hello VoyageAI!")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
#### Check `voyage-3-large` embeddings in depth

We will experiment with `int8` embedding_type.
"""
logger.info("#### Check `voyage-3-large` embeddings in depth")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load Data
"""
logger.info("#### Load Data")

documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
### With `int8` embedding_type

#### Build index
"""
logger.info("### With `int8` embedding_type")

llm = OllamaFunctionCallingAdapter(
    model="command-nightly",
    #     api_key="<YOUR_OPENAI_API_KEY>",
)
embed_model = VoyageEmbedding(
    voyage_api_key="<YOUR_VOYAGE_API_KEY>",
    model_name="voyage-3-large",
    embedding_type="int8",
)

index = VectorStoreIndex.from_documents(
    documents=documents, embed_model=embed_model
)

"""
#### Build retriever
"""
logger.info("#### Build retriever")

search_query_retriever = index.as_retriever()

search_query_retrieved_nodes = search_query_retriever.retrieve(
    "What happened in the summer of 1995?"
)

for n in search_query_retrieved_nodes:
    display_source_node(n, source_length=2000)

"""
### Text-Image Embeddings

[VoyageAI now support multi-modal embedding model](https://docs.voyageai.com/docs/multimodal-embeddings) where both text and image are in same embedding space.
"""
logger.info("### Text-Image Embeddings")


img = Image.open(
    f"{os.path.dirname(__file__)}/data/images/prometheus_paper_card.png")
plt.imshow(img)

embed_model = VoyageEmbedding(
    voyage_api_key="<YOUR_VOYAGE_API_KEY>",
    model_name="voyage-multimodal-3",
    truncation=False,
)

"""
##### Image Embeddings
"""
logger.info("##### Image Embeddings")

embeddings = embed_model.get_image_embedding(
    f"{os.path.dirname(__file__)}/data/images/prometheus_paper_card.png"
)

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
##### Text Embeddings
"""
logger.info("##### Text Embeddings")

embeddings = embed_model.get_text_embedding("prometheus evaluation model")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

logger.info("\n\n[DONE]", bright=True)

from IPython.display import Markdown, display
from PIL import Image
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.settings import Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cohere import Cohere
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

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/cohereai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# CohereAI Embeddings

Cohere Embed is the first embedding model that natively supports float, int8, binary and ubinary embeddings. 

1. v3 models support all embedding types while v2 models support only `float` embedding type.
2. The default `embedding_type` is `float` with `LlamaIndex`. You can customize it for v3 models using parameter `embedding_type`.

In this notebook, we will demonstrate using `Cohere Embeddings` with different `models`, `input_types` and `embedding_types`.

Refer to their [main blog post](https://txt.cohere.com/int8-binary-embeddings/) for more details on Cohere int8 & binary Embeddings.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# CohereAI Embeddings")

# %pip install llama-index-llms-cohere
# %pip install llama-index-embeddings-cohere

# !pip install llama-index


cohere_api_key = "YOUR COHERE API KEY"
os.environ["COHERE_API_KEY"] = cohere_api_key

"""
#### With latest `embed-english-v3.0` embeddings.

- input_type="search_document": Use this for texts (documents) you want to store in your vector database

- input_type="search_query": Use this for search queries to find the most relevant documents in your vector database

The default `embedding_type` is `float`.
"""
logger.info("#### With latest `embed-english-v3.0` embeddings.")


embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

embeddings = embed_model.get_text_embedding("Hello CohereAI!")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_document",
)

embeddings = embed_model.get_text_embedding("Hello CohereAI!")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
##### Let's check With `int8` embedding_type
"""
logger.info("##### Let's check With `int8` embedding_type")

embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",
    embedding_type="int8",
)

embeddings = embed_model.get_text_embedding("Hello CohereAI!")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
##### With `binary` embedding_type
"""
logger.info("##### With `binary` embedding_type")

embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",
    embedding_type="binary",
)

embeddings = embed_model.get_text_embedding("Hello CohereAI!")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
#### With old `embed-english-v2.0` embeddings.

v2 models support by default `float` embedding_type.
"""
logger.info("#### With old `embed-english-v2.0` embeddings.")

embed_model = CohereEmbedding(
    api_key=cohere_api_key, model_name="embed-english-v2.0"
)

embeddings = embed_model.get_text_embedding("Hello CohereAI!")

logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
#### Now with latest `embed-english-v3.0` embeddings, 

let's use 
1. input_type=`search_document` to build index
2. input_type=`search_query` to retrive relevant context.

We will experiment with `int8` embedding_type.
"""
logger.info("#### Now with latest `embed-english-v3.0` embeddings,")


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

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
### With `int8` embedding_type

#### Build index with input_type = 'search_document'
"""
logger.info("### With `int8` embedding_type")

llm = Cohere(model="command-nightly", api_key=cohere_api_key)
embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_document",
    embedding_type="int8",
)

index = VectorStoreIndex.from_documents(
    documents=documents, embed_model=embed_model
)

"""
#### Build retriever with input_type = 'search_query'
"""
logger.info("#### Build retriever with input_type = 'search_query'")

embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",
    embedding_type="int8",
)

search_query_retriever = index.as_retriever()

search_query_retrieved_nodes = search_query_retriever.retrieve(
    "What happened in the summer of 1995?"
)

for n in search_query_retrieved_nodes:
    display_source_node(n, source_length=2000)

"""
### With `float` embedding_type

#### Build index with input_type = 'search_document'
"""
logger.info("### With `float` embedding_type")

llm = Cohere(model="command-nightly", api_key=cohere_api_key)
embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_document",
    embedding_type="float",
)

index = VectorStoreIndex.from_documents(
    documents=documents, embed_model=embed_model
)

"""
#### Build retriever with input_type = 'search_query'
"""
logger.info("#### Build retriever with input_type = 'search_query'")

embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",
    embedding_type="float",
)

search_query_retriever = index.as_retriever()

search_query_retrieved_nodes = search_query_retriever.retrieve(
    "What happened in the summer of 1995?"
)

for n in search_query_retrieved_nodes:
    display_source_node(n, source_length=2000)

"""
### With `binary` embedding_type.

#### Build index with input_type = 'search_document'
"""
logger.info("### With `binary` embedding_type.")

embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_document",
    embedding_type="binary",
)

index = VectorStoreIndex.from_documents(
    documents=documents, embed_model=embed_model
)

"""
#### Build retriever with input_type = 'search_query'
"""
logger.info("#### Build retriever with input_type = 'search_query'")

embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",
    embedding_type="binary",
)

search_query_retriever = index.as_retriever()

search_query_retrieved_nodes = search_query_retriever.retrieve(
    "What happened in the summer of 1995?"
)

for n in search_query_retrieved_nodes:
    display_source_node(n, source_length=2000)

"""
##### The retrieved chunks are certainly different with `binary` embedding type compared to `float` and `int8`. It would be interesting to do [retrieval evaluation](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/usage_pattern_retrieval/) for your RAG pipeline in using `float`/`int8`/`binary`/`ubinary` embeddings.

### Text-Image Embeddings

[Cohere now support multi-modal embedding model](https://cohere.com/blog/multimodal-embed-3) where both text and image are in same embedding space.
"""
logger.info("##### The retrieved chunks are certainly different with `binary` embedding type compared to `float` and `int8`. It would be interesting to do [retrieval evaluation](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/usage_pattern_retrieval/) for your RAG pipeline in using `float`/`int8`/`binary`/`ubinary` embeddings.")


img = Image.open("../data/images/prometheus_paper_card.png")
plt.imshow(img)


embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
)

"""
##### Image Embeddings
"""
logger.info("##### Image Embeddings")

embeddings = embed_model.get_image_embedding(
    "../data/images/prometheus_paper_card.png"
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
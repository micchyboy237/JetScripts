from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import settings
from llama_index.embeddings.nomic import NomicEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/nomic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Nomic Embedding

Nomic has released v1.5 🪆🪆🪆 is capable of variable sized embeddings with matryoshka learning and an 8192 context, embedding dimensions between 64 and 768.

In this notebook, we will explore using Nomic v1.5 embedding at different dimensions.

### Installation
"""
logger.info("# Nomic Embedding")

# %pip install -U llama-index llama-index-embeddings-nomic

"""
### Setup API Keys
"""
logger.info("### Setup API Keys")

nomic_api_key = "<NOMIC API KEY>"

# import nest_asyncio

# nest_asyncio.apply()


"""
#### With dimension at 128
"""
logger.info("#### With dimension at 128")

embed_model = NomicEmbedding(
    api_key=nomic_api_key,
    dimensionality=128,
    model_name="nomic-embed-text-v1.5",
)

embedding = embed_model.get_text_embedding("Nomic Embeddings")

logger.debug(len(embedding))

embedding[:5]

"""
#### With dimension at 256
"""
logger.info("#### With dimension at 256")

embed_model = NomicEmbedding(
    api_key=nomic_api_key,
    dimensionality=256,
    model_name="nomic-embed-text-v1.5",
)

embedding = embed_model.get_text_embedding("Nomic Embeddings")

logger.debug(len(embedding))

embedding[:5]

"""
#### With dimension at 768
"""
logger.info("#### With dimension at 768")

embed_model = NomicEmbedding(
    api_key=nomic_api_key,
    dimensionality=768,
    model_name="nomic-embed-text-v1.5",
)

embedding = embed_model.get_text_embedding("Nomic Embeddings")

logger.debug(len(embedding))

embedding[:5]

"""
#### You can still use v1 Nomic Embeddings

It has 768 fixed embedding dimensions
"""
logger.info("#### You can still use v1 Nomic Embeddings")

embed_model = NomicEmbedding(
    api_key=nomic_api_key, model_name="nomic-embed-text-v1"
)

embedding = embed_model.get_text_embedding("Nomic Embeddings")

logger.debug(len(embedding))

embedding[:5]

"""
### Let's Build end to end RAG pipeline with Nomic v1.5 Embedding.

We will use OllamaFunctionCalling for Generation step.

#### Set Embedding model and llm.
"""
logger.info("### Let's Build end to end RAG pipeline with Nomic v1.5 Embedding.")


# os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"

embed_model = NomicEmbedding(
    api_key=nomic_api_key,
    dimensionality=128,
    model_name="nomic-embed-text-v1.5",
)

llm = OllamaFunctionCalling(model="llama3.2")

settings.llm = llm
settings.embed_model = embed_model

"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load data
"""
logger.info("#### Load data")

documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()

"""
#### Index creation
"""
logger.info("#### Index creation")

index = VectorStoreIndex.from_documents(documents)

"""
#### Query Engine
"""
logger.info("#### Query Engine")

query_engine = index.as_query_engine()

response = query_engine.query("what did author do growing up?")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)

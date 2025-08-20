from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/mixedbread_reranker.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# mixedbread Rerank Cookbook

mixedbread.ai has released three fully open-source reranker models under the Apache 2.0 license. For more in-depth information, you can check out their detailed [blog post](https://www.mixedbread.ai/blog/mxbai-rerank-v1). The following are the three models:

1. `mxbai-rerank-xsmall-v1`
2. `mxbai-rerank-base-v1`
3. `mxbai-rerank-large-v1`

In this notebook, we'll demonstrate how to use the `mxbai-rerank-base-v1` model with the `SentenceTransformerRerank` module in LlamaIndex. This setup allows you to seamlessly swap in any reranker model of your choice using the `SentenceTransformerRerank` module to enhance your RAG pipeline.

### Installation
"""
logger.info("# mixedbread Rerank Cookbook")

# !pip install llama-index
# !pip install sentence-transformers

"""
### Set API Keys
"""
logger.info("### Set API Keys")


# os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"



"""
### Download Data
"""
logger.info("### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Load Documents
"""
logger.info("### Load Documents")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
### Build Index
"""
logger.info("### Build Index")

index = VectorStoreIndex.from_documents(documents=documents)

"""
### Define postprocessor for `mxbai-rerank-base-v1` reranker
"""
logger.info("### Define postprocessor for `mxbai-rerank-base-v1` reranker")


postprocessor = SentenceTransformerRerank(
    model="mixedbread-ai/mxbai-rerank-base-v1", top_n=2
)

"""
### Create Query Engine

We will first retrieve 10 relevant nodes and pick top-2 nodes using the defined postprocessor.
"""
logger.info("### Create Query Engine")

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[postprocessor],
)

"""
### Test Queries
"""
logger.info("### Test Queries")

response = query_engine.query(
    "Why did Sam Altman decline the offer of becoming president of Y Combinator?",
)

logger.debug(response)

response = query_engine.query(
    "Why did Paul Graham start YC?",
)

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
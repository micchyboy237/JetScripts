from IPython.display import Markdown, display
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import UnstructuredReader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Dense-X-Retrieval Pack

This notebook walks through using the `DenseXRetrievalPack`, which parses documents into nodes, and then generates propositions from each node to assist with retreival.

This follows the idea from the paper [Dense X Retrieval: What Retreival Granularity Should We Use?](https://arxiv.org/abs/2312.06648).

From the paper, a proposition is described as:

```
Propositions are defined as atomic expressions within text, each encapsulating a distinct factoid and presented in a concise, self-contained natural language format.
```

We use the provided MLX prompt from their paper to generate propositions, which are then embedded and used to retrieve their parent node chunks.

## Setup
"""
logger.info("# Dense-X-Retrieval Pack")

# %pip install llama-index-llms-ollama
# %pip install llama-index-readers-file


# os.environ["OPENAI_API_KEY"] = "sk-..."

# import nest_asyncio

# nest_asyncio.apply()

# !mkdir -p 'data/'
# !curl 'https://arxiv.org/pdf/2307.09288.pdf' -o 'data/llama2.pdf'


documents = UnstructuredReader().load_data(f"{GENERATED_DIR}/llama2.pdf")

"""
## Run the DenseXRetrievalPack

The `DenseXRetrievalPack` creates both a retriver and query engine.
"""
logger.info("## Run the DenseXRetrievalPack")


DenseXRetrievalPack = download_llama_pack("DenseXRetrievalPack", "./dense_pack")


dense_pack = DenseXRetrievalPack(
    documents,
    proposition_llm=MLXLlamaIndexLLMAdapter(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", max_tokens=750),
    query_llm=MLXLlamaIndexLLMAdapter(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", max_tokens=256),
    text_splitter=SentenceSplitter(chunk_size=1024),
)
dense_query_engine = dense_pack.query_engine


base_index = VectorStoreIndex.from_documents(documents)
base_query_engine = base_index.as_query_engine()

"""
## Test Queries

### How was Llama2 pretrained?
"""
logger.info("## Test Queries")


response = dense_query_engine.query("How was Llama2 pretrained?")
display(Markdown(str(response)))

response = base_query_engine.query("How was Llama2 pretrained?")
display(Markdown(str(response)))

"""
### What baselines are used to compare performance and accuracy?
"""
logger.info("### What baselines are used to compare performance and accuracy?")

response = dense_query_engine.query(
    "What baselines are used to compare performance and accuracy?"
)
display(Markdown(str(response)))

response = base_query_engine.query(
    "What baselines are used to compare performance and accuracy?"
)
display(Markdown(str(response)))

"""
### What datasets were used for measuring performance and accuracy?
"""
logger.info("### What datasets were used for measuring performance and accuracy?")

response = dense_query_engine.query(
    "What datasets were used for measuring performance and accuracy?"
)
display(Markdown(str(response)))

response = base_query_engine.query(
    "What datasets were used for measuring performance and accuracy?"
)
display(Markdown(str(response)))

logger.info("\n\n[DONE]", bright=True)
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/usecases/10k_sub_question.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 10K Analysis
In this demo, we explore answering complex queries by decomposing them into simpler sub-queries.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# 10K Analysis")

# %pip install llama-index-llms-ollama

# !pip install llama-index

# import nest_asyncio

# nest_asyncio.apply()



"""
## Configure LLM service
"""
logger.info("## Configure LLM service")


# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"


Settings.llm = MLXLlamaIndexLLMAdapter(temperature=0.2, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

"""
## Load data
"""
logger.info("## Load data")

lyft_docs = SimpleDirectoryReader(
    input_files=["./data/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()

"""
## Build indices
"""
logger.info("## Build indices")

lyft_index = VectorStoreIndex.from_documents(lyft_docs)

uber_index = VectorStoreIndex.from_documents(uber_docs)

"""
## Build query engines
"""
logger.info("## Build query engines")

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)

uber_engine = uber_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021"
            ),
        ),
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)

"""
## Run queries
"""
logger.info("## Run queries")

response = s_engine.query(
    "Compare and contrast the customer segments and geographies that grew the"
    " fastest"
)

logger.debug(response)

response = s_engine.query(
    "Compare revenue growth of Uber and Lyft from 2020 to 2021"
)

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
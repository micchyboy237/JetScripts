from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer
from typing import Sequence, List
import json
import os
import shutil
import torch


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Llama3 Cookbook

Meta developed and released the Meta [Llama 3](https://ai.meta.com/blog/meta-llama-3/) family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes. The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks.

In this notebook, we will demonstrate how to use Llama3 with LlamaIndex. Here, we use `Llama-3-8B-Instruct` for the demonstration."

### Installation
"""
logger.info("# Llama3 Cookbook")

# !pip install llama-index
# !pip install llama-index-llms-huggingface
# !pip install llama-index-embeddings-huggingface
# !pip install llama-index-embeddings-huggingface-api

"""
To use llama3 from the official repo, you'll need to authorize your huggingface account and use your huggingface token.
"""
logger.info("To use llama3 from the official repo, you'll need to authorize your huggingface account and use your huggingface token.")

hf_token = "hf_"

"""
### Setup Tokenizer and Stopping ids
"""
logger.info("### Setup Tokenizer and Stopping ids")


tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=hf_token,
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

"""
### Setup LLM using `HuggingFaceLLM`
"""
logger.info("### Setup LLM using `HuggingFaceLLM`")


llm = HuggingFaceLLM(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={
        "token": hf_token,
        "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    },
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer_kwargs={"token": hf_token},
    stopping_ids=stopping_ids,
)


"""
### Call complete with a prompt
"""
logger.info("### Call complete with a prompt")

response = llm.complete("Who is Paul Graham?")

logger.debug(response)

"""
### Call chat with a list of messages
"""
logger.info("### Call chat with a list of messages")


messages = [
    ChatMessage(role="system", content="You are CEO of MetaAI"),
    ChatMessage(role="user", content="Introduce Llama3 to the world."),
]
response = llm.chat(messages)

logger.debug(response)

"""
### Let's build RAG pipeline with Llama3

### Download Data
"""
logger.info("### Let's build RAG pipeline with Llama3")

# !wget "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt" "paul_graham_essay.txt"

"""
### Load Data
"""
logger.info("### Load Data")


documents = SimpleDirectoryReader(
    input_files=["paul_graham_essay.txt"]
).load_data()

"""
### Setup Embedding Model
"""
logger.info("### Setup Embedding Model")


embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)

"""
### Set Default LLM and Embedding Model
"""
logger.info("### Set Default LLM and Embedding Model")


Settings.embed_model = embed_model

Settings.llm = llm

"""
### Create Index
"""
logger.info("### Create Index")

index = VectorStoreIndex.from_documents(
    documents,
)

"""
### Create QueryEngine
"""
logger.info("### Create QueryEngine")

query_engine = index.as_query_engine(similarity_top_k=3)

"""
### Querying
"""
logger.info("### Querying")

response = query_engine.query("What did paul graham do growing up?")

logger.debug(response)

"""
### Agents And Tools
"""
logger.info("### Agents And Tools")


# import nest_asyncio

# nest_asyncio.apply()

"""
### Define Tools
"""
logger.info("### Define Tools")


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


def divide(a: int, b: int) -> int:
    """Divides two integers and returns the result integer"""
    return a / b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)

"""
### ReAct Agent
"""
logger.info("### ReAct Agent")

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, subtract_tool, divide_tool],
    llm=llm,
    verbose=True,
)

"""
### Querying
"""
logger.info("### Querying")

response = agent.chat("What is (121 + 2) * 5?")
logger.debug(str(response))

response = agent.chat("What is (100/5)*2-5+10 ?")
logger.debug(str(response))

"""
### ReAct Agent With RAG QueryEngine Tools
"""
logger.info("### ReAct Agent With RAG QueryEngine Tools")


"""
### Download Data
"""
logger.info("### Download Data")

# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'

"""
### Load Data
"""
logger.info("### Load Data")

lyft_docs = SimpleDirectoryReader(
    input_files=[
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=[
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/10k/uber_2021.pdf"]
).load_data()

"""
### Create Indices
"""
logger.info("### Create Indices")

lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)

"""
### Create QueryEngines
"""
logger.info("### Create QueryEngines")

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

"""
### Define QueryEngine Tools
"""
logger.info("### Define QueryEngine Tools")

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

"""
### Create ReAct Agent using RAG QueryEngine Tools
"""
logger.info("### Create ReAct Agent using RAG QueryEngine Tools")

agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
)

"""
### Querying
"""
logger.info("### Querying")

response = agent.chat("What was Lyft's revenue in 2021?")
logger.debug(str(response))

response = agent.chat("What was Uber's revenue in 2021?")
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)

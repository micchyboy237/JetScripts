from IPython.display import Markdown, display
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform.base import (
StepDecomposeQueryTransform,
)
from llama_index.core.query_engine import MultiStepQueryEngine
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_transformations/SimpleIndexDemo-multistep.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Multi-Step Query Engine

We have a multi-step query engine that's able to decompose a complex query into sequential subquestions. This
guide walks you through how to set it up!

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Multi-Step Query Engine")

# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load documents, build the VectorStoreIndex
"""
logger.info("#### Load documents, build the VectorStoreIndex")


# os.environ["OPENAI_API_KEY"] = "sk-..."


gpt35 = OllamaFunctionCallingAdapter(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096)

gpt4 = OllamaFunctionCallingAdapter(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096)

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents)

"""
#### Query Index
"""
logger.info("#### Query Index")


step_decompose_transform = StepDecomposeQueryTransform(llm=gpt4, verbose=True)

step_decompose_transform_gpt3 = StepDecomposeQueryTransform(
    llm=gpt35, verbose=True
)

index_summary = "Used to answer questions about the author"


query_engine = index.as_query_engine(llm=gpt4)
query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform,
    index_summary=index_summary,
)
response_gpt4 = query_engine.query(
    "Who was in the first batch of the accelerator program the author"
    " started?",
)

display(Markdown(f"<b>{response_gpt4}</b>"))

sub_qa = response_gpt4.metadata["sub_qa"]
tuples = [(t[0], t[1].response) for t in sub_qa]
logger.debug(tuples)

response_gpt4 = query_engine.query(
    "In which city did the author found his first company, Viaweb?",
)

logger.debug(response_gpt4)

query_engine = index.as_query_engine(llm=gpt35)
query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform_gpt3,
    index_summary=index_summary,
)

response_gpt3 = query_engine.query(
    "In which city did the author found his first company, Viaweb?",
)

logger.debug(response_gpt3)

logger.info("\n\n[DONE]", bright=True)
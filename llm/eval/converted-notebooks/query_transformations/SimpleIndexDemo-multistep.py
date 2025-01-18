from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_transformations/SimpleIndexDemo-multistep.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Multi-Step Query Engine

We have a multi-step query engine that's able to decompose a complex query into sequential subquestions. This
guide walks you through how to set it up!
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
#### Download Data
"""

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load documents, build the VectorStoreIndex
"""

import os

# os.environ["OPENAI_API_KEY"] = "sk-..."

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from jet.llm.ollama import Ollama
from IPython.display import Markdown, display

gpt35 = Ollama(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096)

gpt4 = Ollama(temperature=0, model="llama3.1", request_timeout=300.0, context_window=4096)

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents)

"""
#### Query Index
"""

from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)

step_decompose_transform = StepDecomposeQueryTransform(llm=gpt4, verbose=True)

step_decompose_transform_gpt3 = StepDecomposeQueryTransform(
    llm=gpt35, verbose=True
)

index_summary = "Used to answer questions about the author"

from llama_index.core.query_engine import MultiStepQueryEngine

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
print(tuples)

response_gpt4 = query_engine.query(
    "In which city did the author found his first company, Viaweb?",
)

print(response_gpt4)

query_engine = index.as_query_engine(llm=gpt35)
query_engine = MultiStepQueryEngine(
    query_engine=query_engine,
    query_transform=step_decompose_transform_gpt3,
    index_summary=index_summary,
)

response_gpt3 = query_engine.query(
    "In which city did the author found his first company, Viaweb?",
)

print(response_gpt3)

logger.info("\n\n[DONE]", bright=True)
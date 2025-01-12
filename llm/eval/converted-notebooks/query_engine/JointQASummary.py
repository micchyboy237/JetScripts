"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/JointQASummary.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Joint QA Summary Query Engine
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-llms-ollama

# !pip install llama-index

import nest_asyncio

nest_asyncio.apply()

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
## Download Data
"""

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Load Data
"""

from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/")
documents = reader.load_data()

from llama_index.llms.ollama import Ollama

gpt4 = Ollama(temperature=0, model="llama3.1", request_timeout=300.0, context_window=4096)

chatgpt = Ollama(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096)

from llama_index.core.composability import QASummaryQueryEngineBuilder

query_engine_builder = QASummaryQueryEngineBuilder(
    llm=gpt4,
)
query_engine = query_engine_builder.build_from_documents(documents)

response = query_engine.query(
    "Can you give me a summary of the author's life?",
)

response = query_engine.query(
    "What did the author do growing up?",
)

response = query_engine.query(
    "What did the author do during his time in art school?",
)
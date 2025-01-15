import asyncio
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/response_synthesizers/tree_summarize.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Tree Summarize
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# !pip install llama-index

"""
## Download Data
"""

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Load Data
"""

from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_files=["/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/paul_graham_essay.txt"]
)

docs = reader.load_data()

text = docs[0].text

"""
## Summarize
"""

from llama_index.core.response_synthesizers import TreeSummarize

summarizer = TreeSummarize(verbose=True)

async def run_async_code_70f1f250():
  response = summarizer.get_response("who is Paul Graham?", [text])
  return response
response = asyncio.run(run_async_code_70f1f250())
logger.success(format_json(response))

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
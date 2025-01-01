from llama_index.core.response_synthesizers import Refine
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/response_synthesizers/refine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Refine

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %pip install llama-index-llms-ollama

# !pip install llama-index

# Download Data

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# Load Data


reader = SimpleDirectoryReader(
    input_files=["/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resumepaul_graham_essay.txt"]
)

docs = reader.load_data()

text = docs[0].text

# Summarize


llm = Ollama(model="llama3.2", request_timeout=300.0, context_window=4096)


summarizer = Refine(llm=llm, verbose=True)

response = summarizer.get_response("who is Paul Graham?", [text])

print(response)

logger.info("\n\n[DONE]", bright=True)

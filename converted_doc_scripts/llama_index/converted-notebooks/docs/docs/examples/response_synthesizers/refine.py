from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core.response_synthesizers import Refine
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/response_synthesizers/refine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Refine

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Refine")

# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
### Download Data
"""
logger.info("### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Load Data
"""
logger.info("## Load Data")


reader = SimpleDirectoryReader(
    input_files=[
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/paul_graham_essay.txt"]
)

docs = reader.load_data()

text = docs[0].text

"""
## Summarize
"""
logger.info("## Summarize")


llm = OllamaFunctionCalling(model="llama3.2")


summarizer = Refine(llm=llm, verbose=True)

response = summarizer.get_response("who is Paul Graham?", [text])

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)

from jet.logger import CustomLogger
from llama_index.llms.palm import PaLM
import google.generativeai as palm
import os
import pprint
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/palm.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# PaLM 

In this short notebook, we show how to use the PaLM LLM from Google in LlamaIndex: https://ai.google/discover/palm2/.

We use the `text-bison-001` model by default.

### Setup

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# PaLM")

# %pip install llama-index-llms-palm

# !pip install llama-index

# !pip install -q google-generativeai


palm_api_key = ""

palm.configure(api_key=palm_api_key)

"""
### Define Model
"""
logger.info("### Define Model")

models = [
    m
    for m in palm.list_models()
    if "generateText" in m.supported_generation_methods
]
model = models[0].name
logger.debug(model)

"""
### Start using our `PaLM` LLM abstraction!
"""
logger.info("### Start using our `PaLM` LLM abstraction!")


model = PaLM(api_key=palm_api_key)

model.complete(prompt)

logger.info("\n\n[DONE]", bright=True)
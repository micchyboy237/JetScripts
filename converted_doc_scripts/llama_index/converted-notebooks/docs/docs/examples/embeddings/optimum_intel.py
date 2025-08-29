from jet.logger import CustomLogger
from llama_index.embeddings.huggingface_optimum_intel import IntelEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/huggingface.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Optimized Embedding Model using Optimum-Intel

LlamaIndex has support for loading quantized embedding models for Intel, using the [Optimum-Intel library](https://huggingface.co/docs/optimum/main/en/intel/index). 

Optimized models are smaller and faster, with minimal accuracy loss, see the [documentation](https://huggingface.co/docs/optimum/main/en/intel/optimization_inc) and an [optimization guide](https://huggingface.co/docs/optimum/main/en/intel/optimization_inc) using the IntelLabs/fastRAG library. 

Optimization is based on math instructions in the Xeon® 4th generation or newer processors. 

In order to be able to load and use the quantized models, install the required dependency `pip install optimum[exporters] optimum-intel neural-compressor intel_extension_for_pytorch`. 

Loading is done using the class `IntelEmbedding`; usage is similar to any HuggingFace local embedding model; See example:
"""
logger.info("# Optimized Embedding Model using Optimum-Intel")

# %pip install llama-index-embeddings-huggingface-optimum-intel


embed_model = IntelEmbedding("Intel/bge-small-en-v1.5-rag-int8-static")

embeddings = embed_model.get_text_embedding("Hello World!")
logger.debug(len(embeddings))
logger.debug(embeddings[:5])

logger.info("\n\n[DONE]", bright=True)
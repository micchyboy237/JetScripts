from jet.logger import logger
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import IpexLLMBgeEmbeddings
from langchain_community.embeddings import QuantizedBgeEmbeddings
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# BAAI

>[Beijing Academy of Artificial Intelligence (BAAI) (Wikipedia)](https://en.wikipedia.org/wiki/Beijing_Academy_of_Artificial_Intelligence),
> also known as `Zhiyuan Institute`, is a Chinese non-profit artificial
> intelligence (AI) research laboratory. `BAAI` conducts AI research
> and is dedicated to promoting collaboration among academia and industry,
> as well as fostering top talent and a focus on long-term research on
> the fundamentals of AI technology. As a collaborative hub, BAAI's founding
> members include leading AI companies, universities, and research institutes.


## Embedding Models

### HuggingFaceBgeEmbeddings

>[BGE models on the HuggingFace](https://huggingface.co/BAAI/bge-large-en-v1.5)
> are one of [the best open-source embedding models](https://huggingface.co/spaces/mteb/leaderboard).

See a [usage example](/docs/integrations/text_embedding/bge_huggingface).
"""
logger.info("# BAAI")


"""
### IpexLLMBgeEmbeddings

>[IPEX-LLM](https://github.com/intel-analytics/ipex-llm) is a PyTorch
> library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU,
> discrete GPU such as Arc, Flex and Max) with very low latency.

See a [usage example running model on Intel CPU](/docs/integrations/text_embedding/ipex_llm).
See a [usage example running model on Intel GPU](/docs/integrations/text_embedding/ipex_llm_gpu).
"""
logger.info("### IpexLLMBgeEmbeddings")


"""
### QuantizedBgeEmbeddings

See a [usage example](/docs/integrations/text_embedding/itrex).
"""
logger.info("### QuantizedBgeEmbeddings")


logger.info("\n\n[DONE]", bright=True)
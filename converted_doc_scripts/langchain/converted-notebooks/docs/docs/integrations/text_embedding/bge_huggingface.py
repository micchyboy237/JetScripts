from jet.logger import logger
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
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
# BGE on Hugging Face

>[BGE models on the HuggingFace](https://huggingface.co/BAAI/bge-large-en-v1.5) are one of [the best open-source embedding models](https://huggingface.co/spaces/mteb/leaderboard).
>BGE model is created by the [Beijing Academy of Artificial Intelligence (BAAI)](https://en.wikipedia.org/wiki/Beijing_Academy_of_Artificial_Intelligence). `BAAI` is a private non-profit organization engaged in AI research and development.

This notebook shows how to use `BGE Embeddings` through `Hugging Face`
"""
logger.info("# BGE on Hugging Face")

# %pip install --upgrade --quiet  sentence_transformers


model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

"""
Note that you need to pass `query_instruction=""` for `model_name="BAAI/bge-m3"` see [FAQ BGE M3](https://huggingface.co/BAAI/bge-m3#faq).
"""
logger.info("Note that you need to pass `query_instruction=""` for `model_name="BAAI/bge-m3"` see [FAQ BGE M3](https://huggingface.co/BAAI/bge-m3#faq).")

embedding = hf.embed_query("hi this is harrison")
len(embedding)

logger.info("\n\n[DONE]", bright=True)
from jet.logger import logger
from langchain_community.embeddings import QuantizedBiEncoderEmbeddings
import os
import shutil
import torch


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
# Embedding Documents using Optimized and Quantized Embedders

Embedding all documents using Quantized Embedders.

The embedders are based on optimized models, created by using [optimum-intel](https://github.com/huggingface/optimum-intel.git) and [IPEX](https://github.com/intel/intel-extension-for-pytorch).

Example text is based on [SBERT](https://www.sbert.net/docs/pretrained_cross-encoders.html).
"""
logger.info("# Embedding Documents using Optimized and Quantized Embedders")


model_name = "Intel/bge-small-en-v1.5-rag-int8-static"
encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

model = QuantizedBiEncoderEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
    query_instruction="Represent this sentence for searching relevant passages: ",
)

"""
Let's ask a question, and compare to 2 documents. The first contains the answer to the question, and the second one does not. 

We can check better suits our query.
"""
logger.info("Let's ask a question, and compare to 2 documents. The first contains the answer to the question, and the second one does not.")

question = "How many people live in Berlin?"

documents = [
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin is well known for its museums.",
]

doc_vecs = model.embed_documents(documents)

query_vec = model.embed_query(question)


doc_vecs_torch = torch.tensor(doc_vecs)

query_vec_torch = torch.tensor(query_vec)

query_vec_torch @ doc_vecs_torch.T

"""
We can see that indeed the first one ranks higher.
"""
logger.info("We can see that indeed the first one ranks higher.")

logger.info("\n\n[DONE]", bright=True)
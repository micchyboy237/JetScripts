from bs4 import BeautifulSoup as Soup
from jet.logger import logger
from langchain import hub
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage import InMemoryByteStore, LocalFileStore
from langchain_chroma import Chroma
from langchain_community.document_loaders.recursive_url_loader import (
RecursiveUrlLoader,
)
from langchain_community.embeddings import QuantizedBiEncoderEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from optimum.intel.ipex import IPEXModelForCausalLM
from pathlib import Path
from transformers import AutoTokenizer, pipeline
import langchain
import os
import shutil
import torch
import uuid


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

In this tutorial, we will demo how to build a RAG pipeline, with the embedding for all documents done using Quantized Embedders.

We will use a pipeline that will:

* Create a document collection.
* Embed all documents using Quantized Embedders.
* Fetch relevant documents for our question.
* Run an LLM answer the question.

For more information about optimized models, we refer to [optimum-intel](https://github.com/huggingface/optimum-intel.git) and [IPEX](https://github.com/intel/intel-extension-for-pytorch).

This tutorial is based on the [Langchain RAG tutorial here](https://towardsai.net/p/machine-learning/dense-x-retrieval-technique-in-langchain-and-llamaindex).
"""
logger.info("# Embedding Documents using Optimized and Quantized Embedders")




DOCSTORE_DIR = "."
DOCSTORE_ID_KEY = "doc_id"

"""
Let's first load up this paper, and split into text chunks of size 1000.
"""
logger.info("Let's first load up this paper, and split into text chunks of size 1000.")

loader = RecursiveUrlLoader(
    "https://ar5iv.labs.arxiv.org/html/1706.03762",
    max_depth=2,
    extractor=lambda x: Soup(x, "html.parser").text,
)
data = loader.load()
logger.debug(f"Loaded {len(data)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
logger.debug(f"Split into {len(all_splits)} documents")

"""
In order to embed our documents, we can use the ```QuantizedBiEncoderEmbeddings```, for efficient and fast embedding.
"""
logger.info("In order to embed our documents, we can use the ```QuantizedBiEncoderEmbeddings```, for efficient and fast embedding.")


model_name = "Intel/bge-small-en-v1.5-rag-int8-static"
encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

model_inc = QuantizedBiEncoderEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
    query_instruction="Represent this sentence for searching relevant passages: ",
)

"""
With our embedder in place, let's define our retriever:
"""
logger.info("With our embedder in place, let's define our retriever:")

def get_multi_vector_retriever(
    docstore_id_key: str, collection_name: str, embedding_function: Embeddings
):
    """Create the composed retriever object."""
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    store = InMemoryByteStore()

    return MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=docstore_id_key,
    )


retriever = get_multi_vector_retriever(DOCSTORE_ID_KEY, "multi_vec_store", model_inc)

"""
Next, we divide each chunk into sub-docs:
"""
logger.info("Next, we divide each chunk into sub-docs:")

child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
id_key = "doc_id"
doc_ids = [str(uuid.uuid4()) for _ in all_splits]

sub_docs = []
for i, doc in enumerate(all_splits):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)

"""
Let's write our documents into our new store. This will use our embedder on each document.
"""
logger.info("Let's write our documents into our new store. This will use our embedder on each document.")

retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, all_splits)))

"""
Great! Our retriever is good to go. Let's load up an LLM, that will reason over the retrieved documents:
"""
logger.info("Great! Our retriever is good to go. Let's load up an LLM, that will reason over the retrieved documents:")


model_id = "Intel/neural-chat-7b-v3-3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = IPEXModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, export=True
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

hf = HuggingFacePipeline(pipeline=pipe)

"""
Next, we will load up a prompt for answering questions using retrieved documents:
"""
logger.info("Next, we will load up a prompt for answering questions using retrieved documents:")


prompt = hub.pull("rlm/rag-prompt")

"""
We can now build our pipeline:
"""
logger.info("We can now build our pipeline:")


rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | hf

"""
Excellent! Let's ask it a question.
We will also use a verbose and debug, to check which documents were used by the model to produce the answer.
"""
logger.info("Excellent! Let's ask it a question.")

langchain.verbose = True
langchain.debug = True

llm_res = rag_chain.invoke(
    "What is the first transduction model relying entirely on self-attention?",
)

llm_res

"""
Based on the retrieved documents, the answer is indeed correct :)
"""
logger.info("Based on the retrieved documents, the answer is indeed correct :)")

logger.info("\n\n[DONE]", bright=True)
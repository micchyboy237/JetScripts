from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# RankLLM Reranker

**[RankLLM](https://github.com/castorini/rank_llm)** is a **flexible reranking framework** supporting **listwise, pairwise, and pointwise ranking models**. It includes **RankVicuna, RankZephyr, MonoT5, DuoT5, LiT5, and FirstMistral**, with integration for **FastChat, vLLM, SGLang, and TensorRT-LLM** for efficient inference. RankLLM is optimized for **retrieval and ranking tasks**, leveraging both **open-source LLMs** and proprietary rerankers like **RankGPT and RankGemini**. It supports **batched inference, first-token reranking, and retrieval via BM25 and SPLADE**.

> **Note:** If using the built-in retriever, RankLLM requires **Pyserini, JDK 21, PyTorch, and Faiss** for retrieval functionality.
"""
logger.info("# RankLLM Reranker")

# %pip install --upgrade --quiet rank_llm

# %pip install --upgrade --quiet jet.adapters.langchain.chat_ollama

# %pip install --upgrade --quiet faiss-cpu

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")


def pretty_print_docs(docs):
    logger.debug(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" +
                d.page_content for i, d in enumerate(docs)]
        )
    )


"""
## Set up the base vector store retriever
Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.
"""
logger.info("## Set up the base vector store retriever")


documents = TextLoader(
    "../document_loaders/example_data/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
for idx, text in enumerate(texts):
    text.metadata["id"] = idx

embedding = OllamaEmbeddings(model="nomic-embed-text")
retriever = FAISS.from_documents(
    texts, embedding).as_retriever(search_kwargs={"k": 20})

"""
# Retrieval + RankLLM Reranking (RankZephyr)

Retrieval without reranking
"""
logger.info("# Retrieval + RankLLM Reranking (RankZephyr)")

query = "What was done to Russia?"
docs = retriever.invoke(query)
pretty_print_docs(docs)

"""
RankZephyr performs listwise reranking for improved retrieval quality but requires at least 24GB of VRAM to run efficiently.
"""
logger.info("RankZephyr performs listwise reranking for improved retrieval quality but requires at least 24GB of VRAM to run efficiently.")


torch.cuda.empty_cache()

compressor = RankLLMRerank(top_n=3, model="rank_zephyr")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

del compressor

compressed_docs = compression_retriever.invoke(query)
pretty_print_docs(compressed_docs)

"""
Can be used within a QA pipeline
"""
logger.info("Can be used within a QA pipeline")


llm = ChatOllama(model="llama3.2")

chain = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="llama3.2"), retriever=compression_retriever
)

chain.invoke({"query": query})

"""
# Retrieval + RankLLM Reranking (RankGPT)

Retrieval without reranking
"""
logger.info("# Retrieval + RankLLM Reranking (RankGPT)")

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)

"""
Retrieval + Reranking with RankGPT
"""
logger.info("Retrieval + Reranking with RankGPT")


compressor = RankLLMRerank(top_n=3, model="gpt", gpt_model="llama3.2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(query)
pretty_print_docs(compressed_docs)

"""
You can use this retriever within a QA pipeline
"""
logger.info("You can use this retriever within a QA pipeline")


llm = ChatOllama(model="llama3.2")

chain = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="llama3.2"), retriever=compression_retriever
)

chain.invoke({"query": query})

logger.info("\n\n[DONE]", bright=True)

from jet.adapters.langchain.chat_ollama import Ollama, OllamaEmbeddings
from jet.logger import logger
from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_experimental.autonomous_agents import BabyAGI
from typing import Optional
import faiss
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
# BabyAGI User Guide

This notebook demonstrates how to implement [BabyAGI](https://github.com/yoheinakajima/babyagi/tree/main) by [Yohei Nakajima](https://twitter.com/yoheinakajima). BabyAGI is an AI agent that can generate and pretend to execute tasks based on a given objective.

This guide will help you understand the components to create your own recursive agents.

Although BabyAGI uses specific vectorstores/model providers (Pinecone, Ollama), one of the benefits of implementing it with LangChain is that you can easily swap those out for different options. In this implementation we use a FAISS vectorstore (because it runs locally and is free).

## Install and Import Required Modules
"""
logger.info("# BabyAGI User Guide")



"""
## Connect to the Vector Store

Depending on what vectorstore you use, this step may look different.
"""
logger.info("## Connect to the Vector Store")


embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

"""
### Run the BabyAGI

Now it's time to create the BabyAGI controller and watch it try to accomplish your objective.
"""
logger.info("### Run the BabyAGI")

OBJECTIVE = "Write a weather report for SF today"

llm = Ollama(temperature=0)

verbose = False
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)

baby_agi({"objective": OBJECTIVE})

logger.info("\n\n[DONE]", bright=True)
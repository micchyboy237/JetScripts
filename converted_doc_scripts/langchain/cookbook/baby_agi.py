from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from typing import Optional
from langchain_experimental.autonomous_agents import BabyAGI
from jet.llm.ollama.base_langchain import Ollama, OllamaEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import faiss

initialize_ollama_settings()

"""
# BabyAGI User Guide

This notebook demonstrates how to implement [BabyAGI](https://github.com/yoheinakajima/babyagi/tree/main) by [Yohei Nakajima](https://twitter.com/yoheinakajima). BabyAGI is an AI agent that can generate and pretend to execute tasks based on a given objective.

This guide will help you understand the components to create your own recursive agents.

Although BabyAGI uses specific vectorstores/model providers (Pinecone, Ollama), one of the benefits of implementing it with LangChain is that you can easily swap those out for different options. In this implementation we use a FAISS vectorstore (because it runs locally and is free).
"""

"""
## Install and Import Required Modules
"""


"""
## Connect to the Vector Store

Depending on what vectorstore you use, this step may look different.
"""


embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query,
                    index, InMemoryDocstore({}), {})

"""
### Run the BabyAGI

Now it's time to create the BabyAGI controller and watch it try to accomplish your objective.
"""

OBJECTIVE = "Write a weather report for SF today"

llm = Ollama(temperature=0)

verbose = False
max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
)

baby_agi({"objective": OBJECTIVE})


logger.info("\n\n[DONE]", bright=True)

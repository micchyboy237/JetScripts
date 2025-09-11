from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_permit import PermitRetriever
from langchain_permit.retrievers import PermitEnsembleRetriever
from langchain_permit.retrievers import PermitSelfQueryRetriever
import ChatModelTabs from "@theme/ChatModelTabs"
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
---
sidebar_label: Permit
---

# PermitRetriever

Permit is an access control platform that provides fine-grained, real-time permission management using various models such as RBAC, ABAC, and ReBAC. It enables organizations to enforce dynamic policies across their applications, ensuring that only authorized users can access specific resources.

### Integration details

This notebook illustrates how to integrate [Permit.io](https://permit.io/) permissions into LangChain retrievers.

We provide two custom retrievers:

- PermitSelfQueryRetriever – Uses a self-query approach to parse the user’s natural-language prompt, fetch the user’s permitted resource IDs from Permit, and apply that filter automatically in a vector store search. 
 
- PermitEnsembleRetriever – Combines multiple underlying retrievers (e.g., BM25 + Vector) via LangChain’s EnsembleRetriever, then filters the merged results with Permit.io.

## Setup

Install the package with the command:

```bash
pip install langchain-permit
```

If you want to get automated tracing from individual queries, you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("# PermitRetriever")


"""
### Installation

```bash
pip install langchain-permit
```

#### Environment Variables

```bash
PERMIT_API_KEY=your_api_key
PERMIT_PDP_URL= # or your real deployment
# OPENAI_API_KEY=sk-...
```
- A running Permit PDP. See [Permit docs](https://docs.permit.io/) for details on setting up your policy and container.
- A vector store or multiple retrievers that we can wrap.
"""
logger.info("### Installation")

# %pip install -qU langchain-permit

"""
## Instantiation

### PermitSelfQueryRetriever

#### Basic Explanation

1. Retrieves permitted document IDs from Permit.  

2. Uses an LLM to parse your query and build a “structured filter,” ensuring only docs with those permitted IDs are considered.

#### Basic Usage

```python

# Step 1: Create / load some documents and build a vector store
docs = [...]
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 2: Initialize the retriever
retriever = PermitSelfQueryRetriever(
    pdp_url="...",
    user={"key": "user-123"},
    resource_type="document",
    action="read",
    llm=...,                # Typically a ChatOllama or other LLM
    vectorstore=vectorstore,
    enable_limit=True,      # optional
)

# Step 3: Query
query = "Give me docs about cats"
results = retriever.get_relevant_documents(query)
for doc in results:
    logger.debug(doc.metadata.get("id"), doc.page_content)
```

### PermitEnsembleRetriever

#### Basic Explanation

1. Uses LangChain’s EnsembleRetriever to gather documents from multiple sub-retrievers (e.g., vector-based, BM25, etc.).
2. After retrieving documents, it calls filter_objects on Permit to eliminate any docs the user isn’t allowed to see.

#### Basic Usage

```python

# Suppose we have two child retrievers: bm25_retriever, vector_retriever
...
ensemble_retriever = PermitEnsembleRetriever(
    pdp_url="...",
    user="user_abc",
    action="read",
    resource_type="document",
    retrievers=[bm25_retriever, vector_retriever],
    weights=None
)

docs = ensemble_retriever.get_relevant_documents("Query about cats")
for doc in docs:
    logger.debug(doc.metadata.get("id"), doc.page_content)
```

### Demo Scripts

For more complete demos, check out the `/langchain_permit/examples/demo_scripts` folder:

1. demo_self_query.py – Demonstrates PermitSelfQueryRetriever.
2. demo_ensemble.py – Demonstrates PermitEnsembleRetriever.

Each script shows how to build or load documents, configure Permit, and run queries.

### Conclusion

With these custom retrievers, you can seamlessly integrate Permit.io’s permission checks into LangChain’s retrieval workflow. You can keep your application’s vector search logic while ensuring only authorized documents are returned.

For more details on setting up Permit policies, see the official Permit docs. If you want to combine these with other tools (like JWT validation or a broader RAG pipeline), check out our docs/tools.ipynb in the examples folder.
"""
logger.info("## Instantiation")


retriever = PermitRetriever(
)

"""
## Usage


"""
logger.info("## Usage")

query = "..."

retriever.invoke(query)

"""
## Use within a chain

Like other retrievers, PermitRetriever can be incorporated into LLM applications via [chains](https://docs.permit.io/).

We will need a LLM or chat model:


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Use within a chain")


llm = ChatOllama(model="llama3.2")


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke("...")

"""
## API reference

For detailed documentation of all PermitRetriever features and configurations head to the [Repo](https://github.com/permitio/langchain-permit/tree/master/langchain_permit/examples/demo_scripts).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)

from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from ragatouille import RAGPretrainedModel
import os
import requests
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
# RAGatouille


>[RAGatouille](https://github.com/bclavie/RAGatouille) makes it as simple as can be to use `ColBERT`!
>
>[ColBERT](https://github.com/stanford-futuredata/ColBERT) is a fast and accurate retrieval model, enabling scalable BERT-based search over large text collections in tens of milliseconds.
>
>See the [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) paper.

We can use this as a [retriever](/docs/how_to#retrievers). It will show functionality specific to this integration. After going through, it may be useful to explore [relevant use-case pages](/docs/how_to#qa-with-rag) to learn how to use this vector store as part of a larger chain.

This page covers how to use [RAGatouille](https://github.com/bclavie/RAGatouille) as a retriever in a LangChain chain. 

## Setup

The integration lives in the `ragatouille` package.

```bash
pip install -U ragatouille
```

## Usage

This example is taken from their documentation
"""
logger.info("# RAGatouille")


RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")



def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    URL = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

full_document = get_wikipedia_page("Hayao_Miyazaki")

RAG.index(
    collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True,
)

results = RAG.search(query="What animation studio did Miyazaki found?", k=3)

results

"""
We can then convert easily to a LangChain retriever! We can pass in any kwargs we want when creating (like `k`)
"""
logger.info("We can then convert easily to a LangChain retriever! We can pass in any kwargs we want when creating (like `k`)")

retriever = RAG.as_langchain_retriever(k=3)

retriever.invoke("What animation studio did Miyazaki found?")

"""
## Chaining

We can easily combine this retriever in to a chain.
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
)

llm = ChatOllama(model="llama3.2")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

retrieval_chain.invoke({"input": "What animation studio did Miyazaki found?"})

for s in retrieval_chain.stream({"input": "What animation studio did Miyazaki found?"}):
    logger.debug(s.get("answer", ""), end="")

logger.info("\n\n[DONE]", bright=True)
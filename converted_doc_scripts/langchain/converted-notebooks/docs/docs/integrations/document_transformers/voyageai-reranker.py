from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_voyageai import VoyageAIRerank
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
# VoyageAI Reranker

>[Voyage AI](https://www.voyageai.com/) provides cutting-edge embedding/vectorizations models.

This notebook shows how to use [Voyage AI's rerank endpoint](https://api.voyageai.com/v1/rerank) in a retriever. This builds on top of ideas in the [ContextualCompressionRetriever](/docs/how_to/contextual_compression).
"""
logger.info("# VoyageAI Reranker")

# %pip install --upgrade --quiet  voyageai
# %pip install --upgrade --quiet  langchain-voyageai

# %pip install --upgrade --quiet  faiss


# %pip install --upgrade --quiet  faiss-cpu

# import getpass

if "VOYAGE_API_KEY" not in os.environ:
#     os.environ["VOYAGE_API_KEY"] = getpass.getpass("Voyage AI API Key:")

def pretty_print_docs(docs):
    logger.debug(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

"""
## Set up the base vector store retriever
Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs. You can use any of the following Embeddings models: ([source](https://docs.voyageai.com/docs/embeddings)):

- `voyage-3`
- `voyage-3-lite` 
- `voyage-large-2`
- `voyage-code-2`
- `voyage-2`
- `voyage-law-2`
- `voyage-lite-02-instruct`
- `voyage-finance-2`
- `voyage-multilingual-2`
"""
logger.info("## Set up the base vector store retriever")


documents = TextLoader("../../how_to/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(
    texts, VoyageAIEmbeddings(model="voyage-law-2")
).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)

"""
## Doing reranking with VoyageAIRerank
Now let's wrap our base retriever with a `ContextualCompressionRetriever`. We'll use the Voyage AI reranker to rerank the returned results. You can use any of the following Reranking models: ([source](https://docs.voyageai.com/docs/reranker)):

- `rerank-2`
- `rerank-2-lite`
- `rerank-1`
- `rerank-lite-1`
"""
logger.info("## Doing reranking with VoyageAIRerank")


llm = Ollama(temperature=0)
compressor = VoyageAIRerank(
    model="rerank-lite-1", voyageai_api_key=os.environ["VOYAGE_API_KEY"], top_k=3
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)

"""
You can of course use this retriever within a QA pipeline
"""
logger.info("You can of course use this retriever within a QA pipeline")


chain = RetrievalQA.from_chain_type(
    llm=Ollama(temperature=0), retriever=compression_retriever
)

chain({"query": query})

logger.info("\n\n[DONE]", bright=True)
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# FlashRank reranker

>[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) is the Ultra-lite & Super-fast Python library to add re-ranking to your existing search & retrieval pipelines. It is based on SoTA cross-encoders, with gratitude to all the model owners.

This notebook shows how to use [flashrank](https://github.com/PrithivirajDamodaran/FlashRank) for document compression and retrieval.
"""
logger.info("# FlashRank reranker")

# %pip install --upgrade --quiet  flashrank
# %pip install --upgrade --quiet  faiss


# %pip install --upgrade --quiet  faiss_cpu

def pretty_print_docs(docs):
    logger.debug(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i + 1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(docs)
            ]
        )
    )


"""
## Set up the base vector store retriever
Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.
"""
logger.info("## Set up the base vector store retriever")

# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass()


documents = TextLoader(
    "../../how_to/state_of_the_union.txt",
).load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
for idx, text in enumerate(texts):
    text.metadata["id"] = idx

embedding = OllamaEmbeddings(model="nomic-embed-text")
retriever = FAISS.from_documents(
    texts, embedding).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)

"""
## Doing reranking with FlashRank
Now let's wrap our base retriever with a `ContextualCompressionRetriever`, using `FlashrankRerank` as a compressor.
"""
logger.info("## Doing reranking with FlashRank")


llm = ChatOllama(model="llama3.2")

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
logger.debug([doc.metadata["id"] for doc in compressed_docs])

"""
After reranking, the top 3 documents are different from the top 3 documents retrieved by the base retriever.
"""
logger.info("After reranking, the top 3 documents are different from the top 3 documents retrieved by the base retriever.")

pretty_print_docs(compressed_docs)

"""
## QA reranking with FlashRank
"""
logger.info("## QA reranking with FlashRank")


chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)

chain.invoke(query)

logger.info("\n\n[DONE]", bright=True)

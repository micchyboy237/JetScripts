from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMLinguaCompressor
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
# LLMLingua Document Compressor

>[LLMLingua](https://github.com/microsoft/LLMLingua) utilizes a compact, well-trained language model (e.g., GPT2-small, LLaMA-7B) to identify and remove non-essential tokens in prompts. This approach enables efficient inference with large language models (LLMs), achieving up to 20x compression with minimal performance loss.

This notebook shows how to use LLMLingua as a document compressor.
"""
logger.info("# LLMLingua Document Compressor")

# %pip install --upgrade --quiet  llmlingua accelerate


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
    "../../how_to/state_of_the_union.txt",
).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embedding = OllamaEmbeddings(model="mxbai-embed-large")
retriever = FAISS.from_documents(
    texts, embedding).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.invoke(query)
pretty_print_docs(docs)

"""
## Doing compression with LLMLingua
Now let’s wrap our base retriever with a `ContextualCompressionRetriever`, using `LLMLinguaCompressor` as a compressor.
"""
logger.info("## Doing compression with LLMLingua")


llm = ChatOllama(model="llama3.2")

compressor = LLMLinguaCompressor(
    model_name="ollama-community/gpt2", device_map="cpu")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
pretty_print_docs(compressed_docs)

"""
## QA generation with LLMLingua

We can see what it looks like to use this in the generation step now
"""
logger.info("## QA generation with LLMLingua")


chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)

chain.invoke({"query": query})

logger.info("\n\n[DONE]", bright=True)

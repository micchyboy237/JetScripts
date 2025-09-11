from jet.logger import logger
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together import Together
from langchain_together.embeddings import TogetherEmbeddings
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
## Together AI + RAG
 
[Together AI](https://python.langchain.com/docs/integrations/llms/together) has a broad set of OSS LLMs via inference API.

See [here](https://docs.together.ai/docs/inference-models). We use `"mistralai/Mixtral-8x7B-Instruct-v0.1` for RAG on the Mixtral paper.

Download the paper:
https://arxiv.org/pdf/2401.04088.pdf
"""
logger.info("## Together AI + RAG")

# ! pip install --quiet pypdf tiktoken ollama langchain-chroma langchain-together


loader = PyPDFLoader("~/Desktop/mixtral.pdf")
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


"""
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
"""
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model="mxbai-embed-large"),
)

retriever = vectorstore.as_retriever()


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.0,
    max_tokens=2000,
    top_k=1,
)

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke("What are the Architectural details of Mixtral?")

"""
Trace: 

https://smith.langchain.com/public/935fd642-06a6-4b42-98e3-6074f93115cd/r
"""
logger.info("Trace:")

logger.info("\n\n[DONE]", bright=True)
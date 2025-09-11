from jet.logger import logger
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_fireworks.embeddings import FireworksEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together import Together
import fireworks
import fireworks.client
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
## Fireworks.AI + LangChain + RAG
 
[Fireworks AI](https://python.langchain.com/docs/integrations/llms/fireworks) wants to provide the best experience when working with LangChain, and here is an example of Fireworks + LangChain doing RAG

See [our models page](https://fireworks.ai/models) for the full list of models. We use `accounts/fireworks/models/mixtral-8x7b-instruct` for RAG In this tutorial.

For the RAG target, we will use the Gemma technical report https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf
"""
logger.info("## Fireworks.AI + LangChain + RAG")

# %pip install --quiet pypdf langchain-chroma tiktoken ollama
# %pip uninstall -y langchain-fireworks
# %pip install --editable /mnt/disks/data/langchain/libs/partners/fireworks


logger.debug(fireworks)


url = "https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf"
response = requests.get(url, stream=True)
file_name = "temp_file.pdf"
with open(file_name, "wb") as pdf:
    pdf.write(response.content)

loader = PyPDFLoader(file_name)
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=FireworksEmbeddings(),
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
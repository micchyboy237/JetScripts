from jet.logger import logger
from langchain import hub
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnablePick
import os
import shutil
import sys


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
# RAG application running locally on Intel Xeon CPU using langchain and open-source models

Author - Pratool Bharti (pratool.bharti@intel.com)

In this cookbook, we use langchain tools and open source models to execute locally on CPU. This notebook has been validated to run on Intel Xeon 8480+ CPU. Here we implement a RAG pipeline for Llama2 model to answer questions about Intel Q1 2024 earnings release.

**Create a conda or virtualenv environment with python >=3.10 and install following libraries**
<br>

`pip install --upgrade langchain langchain-community langchainhub langchain-chroma bs4 gpt4all pypdf pysqlite3-binary` <br>
`pip install llama-cpp-python   --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`

**Load pysqlite3 in sys modules since ChromaDB requires sqlite3.**
"""
logger.info("# RAG application running locally on Intel Xeon CPU using langchain and open-source models")

__import__("pysqlite3")

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

"""
**Import essential components from langchain to load and split data**
"""


"""
**Download Intel Q1 2024 earnings release**
"""

# !wget  'https://d1io3yog0oux5.cloudfront.net/_11d435a500963f99155ee058df09f574/intel/db/887/9014/earnings_release/Q1+24_EarningsRelease_FINAL.pdf' -O intel_q1_2024_earnings.pdf

"""
**Loading earning release pdf document through PyPDFLoader**
"""

loader = PyPDFLoader("intel_q1_2024_earnings.pdf")
data = loader.load()

"""
**Splitting entire document in several chunks with each chunk size is 500 tokens**
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

"""
**Looking at the first split of the document**
"""

all_splits[0]

"""
**One of the major step in RAG is to convert each split of document into embeddings and store in a vector database such that searching relevant documents are efficient.** <br>
**For that, importing Chroma vector database from langchain. Also, importing open source GPT4All for embedding models**
"""


"""
**In next step, we will download one of the most popular embedding model "all-MiniLM-L6-v2". Find more details of the model at this link https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2**
"""

model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {"allow_download": "True"}
embeddings = GPT4AllEmbeddings(model_name=model_name, gpt4all_kwargs=gpt4all_kwargs)

"""
**Store all the embeddings in the Chroma database**
"""

vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

"""
**Now, let's find relevant splits from the documents related to the question**
"""

question = "What is Intel CCG revenue in Q1 2024"
docs = vectorstore.similarity_search(question)
logger.debug(len(docs))

"""
**Look at the first retrieved document from the vector database**
"""

docs[0]

"""
**Download Lllama-2 model from Huggingface and store locally** <br>
**You can download different quantization variant of Lllama-2 model from the link below. We are using Q8 version here (7.16GB).** <br>
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
"""
logger.info("https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")

# !huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q8_0.gguf --local-dir . --local-dir-use-symlinks False

"""
**Import langchain components required to load downloaded LLMs model**
"""


"""
**Loading the local Lllama-2 model using Llama-cpp library**
"""

llm = LlamaCpp(
    model_path="llama-2-7b-chat.Q8_0.gguf",
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

"""
**Now let's ask the same question to Llama model without showing them the earnings release.**
"""

llm.invoke(question)

"""
**As you can see, model is giving wrong information. Correct asnwer is CCG revenue in Q1 2024 is $7.5B. Now let's apply RAG using the earning release document**

**in RAG, we modify the input prompt by adding relevent documents with the question. Here, we use one of the popular RAG prompt**
"""


rag_prompt = hub.pull("rlm/rag-prompt")
rag_prompt.messages

"""
*
*
A
p
p
e
n
d
i
n
g
 
a
l
l
 
r
e
t
r
i
e
v
e
d
 
d
o
c
u
m
e
n
t
s
 
i
n
 
a
 
s
i
n
g
l
e
 
d
o
c
u
m
e
n
t
*
*
"""
logger.info("A")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

"""
**The last step is to create a chain using langchain tool that will create an e2e pipeline. It will take question and context as an input.**
"""


chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt
    | llm
    | StrOutputParser()
)

chain.invoke({"context": docs, "question": question})

"""
**Now we see the results are correct as it is mentioned in earnings release.** <br>
**To further automate, we will create a chain that will take input as question and retriever so that we don't need to retrieve documents separately**
"""

retriever = vectorstore.as_retriever()
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

"""
**Now we only need to pass the question to the chain and it will fetch the contexts directly from the vector database to generate the answer**
<br>
**Let's try with another question**
"""

qa_chain.invoke("what is Intel DCAI revenue in Q1 2024?")

logger.info("\n\n[DONE]", bright=True)
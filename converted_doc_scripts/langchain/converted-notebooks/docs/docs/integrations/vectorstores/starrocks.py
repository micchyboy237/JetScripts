from jet.adapters.langchain.chat_ollama import Ollama, OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
DirectoryLoader,
UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import StarRocks
from langchain_community.vectorstores.starrocks import StarRocksSettings
from langchain_text_splitters import TokenTextSplitter
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
# StarRocks

>[StarRocks](https://www.starrocks.io/) is a High-Performance Analytical Database.
`StarRocks` is a next-gen sub-second MPP database for full analytics scenarios, including multi-dimensional analytics, real-time analytics and ad-hoc query.

>Usually `StarRocks` is categorized into OLAP, and it has showed excellent performance in [ClickBench — a Benchmark For Analytical DBMS](https://benchmark.clickhouse.com/). Since it has a super-fast vectorized execution engine, it could also be used as a fast vectordb.

Here we'll show how to use the StarRocks Vector Store.

## Setup
"""
logger.info("# StarRocks")

# %pip install --upgrade --quiet  pymysql langchain-community

"""
Set `update_vectordb = False` at the beginning. If there is no docs updated, then we don't need to rebuild the embeddings of docs
"""
logger.info("Set `update_vectordb = False` at the beginning. If there is no docs updated, then we don't need to rebuild the embeddings of docs")


update_vectordb = False

"""
## Load docs and split them into tokens

Load all markdown files under the `docs` directory

for starrocks documents, you can clone repo from https://github.com/StarRocks/starrocks, and there is `docs` directory in it.
"""
logger.info("## Load docs and split them into tokens")

loader = DirectoryLoader(
    "./docs", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
)
documents = loader.load()

"""
Split docs into tokens, and set `update_vectordb = True` because there are new docs/tokens.
"""
logger.info("Split docs into tokens, and set `update_vectordb = True` because there are new docs/tokens.")

text_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

update_vectordb = True

split_docs[-20]

logger.debug("# docs  = %d, # splits = %d" % (len(documents), len(split_docs)))

"""
## Create vectordb instance

### Use StarRocks as vectordb
"""
logger.info("## Create vectordb instance")

def gen_starrocks(update_vectordb, embeddings, settings):
    if update_vectordb:
        docsearch = StarRocks.from_documents(split_docs, embeddings, config=settings)
    else:
        docsearch = StarRocks(embeddings, settings)
    return docsearch

"""
## Convert tokens into embeddings and put them into vectordb

Here we use StarRocks as vectordb, you can configure StarRocks instance via `StarRocksSettings`.

Configuring StarRocks instance is pretty much like configuring mysql instance. You need to specify:
1. host/port
2. username(default: 'root')
3. password(default: '')
4. database(default: 'default')
5. table(default: 'langchain')
"""
logger.info("## Convert tokens into embeddings and put them into vectordb")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

settings = StarRocksSettings()
settings.port = 41003
settings.host = "127.0.0.1"
settings.username = "root"
settings.password = ""
settings.database = "zya"
docsearch = gen_starrocks(update_vectordb, embeddings, settings)

logger.debug(docsearch)

update_vectordb = False

"""
## Build QA and ask question to it
"""
logger.info("## Build QA and ask question to it")

llm = Ollama()
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)
query = "is profile enabled by default? if not, how to enable profile?"
resp = qa.run(query)
logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)
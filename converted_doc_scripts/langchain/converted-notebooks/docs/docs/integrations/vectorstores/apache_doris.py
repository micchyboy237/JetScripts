from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores.apache_doris import (
    ApacheDoris,
    ApacheDorisSettings,
)
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
# Apache Doris

>[Apache Doris](https://doris.apache.org/) is a modern data warehouse for real-time analytics.
It delivers lightning-fast analytics on real-time data at scale.

>Usually `Apache Doris` is categorized into OLAP, and it has showed excellent performance in [ClickBench — a Benchmark For Analytical DBMS](https://benchmark.clickhouse.com/). Since it has a super-fast vectorized execution engine, it could also be used as a fast vectordb.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

Here we'll show how to use the Apache Doris Vector Store.

## Setup
"""
logger.info("# Apache Doris")

# %pip install --upgrade --quiet  pymysql

"""
Set `update_vectordb = False` at the beginning. If there is no docs updated, then we don't need to rebuild the embeddings of docs
"""
logger.info("Set `update_vectordb = False` at the beginning. If there is no docs updated, then we don't need to rebuild the embeddings of docs")

# !pip install  sqlalchemy
# !pip install langchain


update_vectordb = False

"""
## Load docs and split them into tokens

Load all markdown files under the `docs` directory

for Apache Doris documents, you can clone repo from https://github.com/apache/doris, and there is `docs` directory in it.
"""
logger.info("## Load docs and split them into tokens")

loader = DirectoryLoader(
    "./docs", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
)
documents = loader.load()

"""
Split docs into tokens, and set `update_vectordb = True` because there are new docs/tokens.
"""
logger.info(
    "Split docs into tokens, and set `update_vectordb = True` because there are new docs/tokens.")

text_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

update_vectordb = True

"""
split_docs[-20]

logger.debug("# docs  = %d, # splits = %d" % (len(documents), len(split_docs)))

## Create vectordb instance

### Use Apache Doris as vectordb
"""
logger.info("## Create vectordb instance")


def gen_apache_doris(update_vectordb, embeddings, settings):
    if update_vectordb:
        docsearch = ApacheDoris.from_documents(
            split_docs, embeddings, config=settings)
    else:
        docsearch = ApacheDoris(embeddings, settings)
    return docsearch


"""
## Convert tokens into embeddings and put them into vectordb

Here we use Apache Doris as vectordb, you can configure Apache Doris instance via `ApacheDorisSettings`.

Configuring Apache Doris instance is pretty much like configuring mysql instance. You need to specify:
1. host/port
2. username(default: 'root')
3. password(default: '')
4. database(default: 'default')
5. table(default: 'langchain')
"""
logger.info("## Convert tokens into embeddings and put them into vectordb")

# from getpass import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass()

update_vectordb = True

embeddings = OllamaEmbeddings(model="nomic-embed-text")

settings = ApacheDorisSettings()
settings.port = 9030
settings.host = "172.30.34.130"
settings.username = "root"
settings.password = ""
settings.database = "langchain"
docsearch = gen_apache_doris(update_vectordb, embeddings, settings)

logger.debug(docsearch)

update_vectordb = False

"""
## Build QA and ask question to it
"""
logger.info("## Build QA and ask question to it")

llm = ChatOllama()
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)
query = "what is apache doris"
resp = qa.run(query)
logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)

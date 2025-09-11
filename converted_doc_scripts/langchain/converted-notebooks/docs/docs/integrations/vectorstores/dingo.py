from dingodb import DingoDB
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Dingo
from langchain_text_splitters import CharacterTextSplitter
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
# DingoDB

>[DingoDB](https://dingodb.readthedocs.io/en/latest/) is a distributed multi-mode vector database, which combines the characteristics of data lakes and vector databases, and can store data of any type and size (Key-Value, PDF, audio, video, etc.). It has real-time low-latency processing capabilities to achieve rapid insight and response, and can efficiently conduct instant analysis and process multi-modal data.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use functionality related to the DingoDB vector database.

To run, you should have a [DingoDB instance up and running](https://github.com/dingodb/dingo-deploy/blob/main/README.md).
"""
logger.info("# DingoDB")

# %pip install --upgrade --quiet  dingodb
# %pip install --upgrade --quiet  git+https://git@github.com/dingodb/pydingo.git

"""
We want to use OllamaEmbeddings so we have to get the Ollama API Key.
"""
logger.info(
    "We want to use OllamaEmbeddings so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")


index_name = "langchain_demo"

dingo_client = DingoDB(user="", password="", host=["127.0.0.1:13000"])
if (
    index_name not in dingo_client.get_index()
    and index_name.upper() not in dingo_client.get_index()
):
    dingo_client.create_index(
        index_name=index_name, dimension=1536, metric_type="cosine", auto_id=False
    )

docsearch = Dingo.from_documents(
    docs, embeddings, client=dingo_client, index_name=index_name
)


query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)

logger.debug(docs[0].page_content)

"""
### Adding More Text to an Existing Index

More text can embedded and upserted to an existing Dingo index using the `add_texts` function
"""
logger.info("### Adding More Text to an Existing Index")

vectorstore = Dingo(embeddings, "text", client=dingo_client,
                    index_name=index_name)

vectorstore.add_texts(["More text!"])

"""
### Maximal Marginal Relevance Searches

In addition to using similarity search in the retriever object, you can also use `mmr` as retriever.
"""
logger.info("### Maximal Marginal Relevance Searches")

retriever = docsearch.as_retriever(search_type="mmr")
matched_docs = retriever.invoke(query)
for i, d in enumerate(matched_docs):
    logger.debug(f"\n## Document {i}\n")
    logger.debug(d.page_content)

"""
Or use `max_marginal_relevance_search` directly:
"""
logger.info("Or use `max_marginal_relevance_search` directly:")

found_docs = docsearch.max_marginal_relevance_search(query, k=2, fetch_k=10)
for i, doc in enumerate(found_docs):
    logger.debug(f"{i + 1}.", doc.page_content, "\n")

logger.info("\n\n[DONE]", bright=True)

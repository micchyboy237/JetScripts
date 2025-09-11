from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import USearch
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
# USearch
>[USearch](https://unum-cloud.github.io/usearch/) is a Smaller & Faster Single-File Vector Search Engine

>USearch's base functionality is identical to FAISS, and the interface should look familiar if you have ever investigated Approximate Nearest Neigbors search. FAISS is a widely recognized standard for high-performance vector search engines. USearch and FAISS both employ the same HNSW algorithm, but they differ significantly in their design principles. USearch is compact and broadly compatible without sacrificing performance, with a primary focus on user-defined metrics and fewer dependencies.
"""
logger.info("# USearch")

# %pip install --upgrade --quiet  usearch langchain-community

"""
We want to use OllamaEmbeddings so we have to get the Ollama API Key.
"""
logger.info(
    "We want to use OllamaEmbeddings so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")


loader = TextLoader("../../../extras/modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = USearch.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

logger.debug(docs[0].page_content)

"""
## Similarity Search with score
The `similarity_search_with_score` method allows you to return not only the documents but also the distance score of the query to them. The returned distance score is L2 distance. Therefore, a lower score is better.
"""
logger.info("## Similarity Search with score")

docs_and_scores = db.similarity_search_with_score(query)

docs_and_scores[0]

logger.info("\n\n[DONE]", bright=True)

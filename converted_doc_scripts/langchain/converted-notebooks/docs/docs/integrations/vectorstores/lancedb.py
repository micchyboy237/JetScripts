from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from lancedb.rerankers import LinearCombinationReranker
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import LanceDB
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_text_splitters import CharacterTextSplitter
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
# LanceDB

>[LanceDB](https://lancedb.com/) is an open-source database for vector-search built with persistent storage, which greatly simplifies retrevial, filtering and management of embeddings. Fully open source.

This notebook shows how to use functionality related to the `LanceDB` vector database based on the Lance data format.
"""
logger.info("# LanceDB")

# ! pip install tantivy

# ! pip install -U langchain-ollama langchain-community

# ! pip install lancedb

"""
We want to use OllamaEmbeddings so we have to get the Ollama API Key.
"""
logger.info("We want to use OllamaEmbeddings so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

# ! rm -rf /tmp/lancedb


loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()

documents = CharacterTextSplitter().split_documents(documents)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

"""
##### For LanceDB cloud, you can invoke the vector store as follows :


```python
db_url = "db://lang_test" # url of db you created
 # your API key
region="us-east-1-dev"  # your selected region

vector_store = LanceDB(
    uri=db_url,
    api_key=api_key,
    region=region,
    embedding=embeddings,
    table_name='langchain_test'
    )
```

You can also add `region`, `api_key`, `uri` to `from_documents()` classmethod
"""
logger.info("##### For LanceDB cloud, you can invoke the vector store as follows :")


reranker = LinearCombinationReranker(weight=0.3)

docsearch = LanceDB.from_documents(documents, embeddings, reranker=reranker)
query = "What did the president say about Ketanji Brown Jackson"

docs = docsearch.similarity_search_with_relevance_scores(query)
logger.debug("relevance score - ", docs[0][1])
logger.debug("text- ", docs[0][0].page_content[:1000])

docs = docsearch.similarity_search_with_score(query="Headaches", query_type="hybrid")
logger.debug("distance - ", docs[0][1])
logger.debug("text- ", docs[0][0].page_content[:1000])

logger.debug("reranker : ", docsearch._reranker)

"""
Additionaly, to explore the table you can load it into a df or save it in a csv file: 
```python
tbl = docsearch.get_table()
logger.debug("tbl:", tbl)
pd_df = tbl.to_pandas()
# pd_df.to_csv("docsearch.csv", index=False)

# you can also create a new vector store object using an older connection object:
vector_store = LanceDB(connection=tbl, embedding=embeddings)
```
"""
logger.info("# pd_df.to_csv("docsearch.csv", index=False)")

docs = docsearch.similarity_search(
    query=query, filter={"metadata.source": "../../how_to/state_of_the_union.txt"}
)

logger.debug("metadata :", docs[0].metadata)


logger.debug("\nSQL filtering :\n")
docs = docsearch.similarity_search(query=query, filter="text LIKE '%Officer Rivera%'")
logger.debug(docs[0].page_content)

"""
## Adding images
"""
logger.info("## Adding images")

# ! pip install -U langchain-experimental

# ! pip install open_clip_torch torch

# ! rm -rf '/tmp/multimmodal_lance'




image_urls = [
    "https://github.com/raghavdixit99/assets/assets/34462078/abf47cc4-d979-4aaa-83be-53a2115bf318",
    "https://github.com/raghavdixit99/assets/assets/34462078/93be928e-522b-4e37-889d-d4efd54b2112",
]

texts = ["bird", "dragon"]

dir_name = "./photos/"

os.makedirs(dir_name, exist_ok=True)

image_uris = []
for i, url in enumerate(image_urls, start=1):
    response = requests.get(url)
    path = os.path.join(dir_name, f"image{i}.jpg")
    image_uris.append(path)
    with open(path, "wb") as f:
        f.write(response.content)


vec_store = LanceDB(
    table_name="multimodal_test",
    embedding=OpenCLIPEmbeddings(),
)

vec_store.add_images(uris=image_uris)

vec_store.add_texts(texts)

img_embed = vec_store._embedding.embed_query("bird")

vec_store.similarity_search_by_vector(img_embed)[0]

vec_store._table

logger.info("\n\n[DONE]", bright=True)
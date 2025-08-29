from IPython.display import Markdown, display
from PIL import Image
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.embeddings.jinaai import JinaEmbedding
from numpy import dot
from numpy.linalg import norm
import logging
import os
import requests
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/jinaai_embeddings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Jina Embeddings

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Jina Embeddings")

# %pip install llama-index-embeddings-jinaai
# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
You may also need other packages that do not come direcly with llama-index
"""
logger.info("You may also need other packages that do not come direcly with llama-index")

# !pip install Pillow

"""
For this example, you will need an API key which you can get from https://jina.ai/embeddings/
"""
logger.info("For this example, you will need an API key which you can get from https://jina.ai/embeddings/")


jinaai_api_key = "YOUR_JINAAI_API_KEY"
os.environ["JINAAI_API_KEY"] = jinaai_api_key

"""
## Embed text and queries with Jina embedding models through JinaAI API

You can encode your text and your queries using the JinaEmbedding class. Jina offers a range of models adaptable to various use cases.

|  Model | Dimension  |  Language |  MRL (matryoshka) | Context |
|:----------------------:|:---------:|:---------:|:-----------:|:---------:|
|  jina-embeddings-v3  |  1024 | Multilingual (89 languages)  |  Yes  | 8192 |
|  jina-embeddings-v2-base-en |  768 |  English |  No | 8192  | 
|  jina-embeddings-v2-base-de |  768 |  German & English |  No  |  8192 | 
|  jina-embeddings-v2-base-es |  768 |  Spanish & English |  No  |  8192 | 
|  jina-embeddings-v2-base-zh | 768  |  Chinese & English |  No  |  8192 | 

**Recommended Model: jina-embeddings-v3 :**

We recommend `jina-embeddings-v3` as the latest and most performant embedding model from Jina AI. This model features 5 task-specific adapters trained on top of its backbone, optimizing various embedding use cases.

By default `JinaEmbedding` class uses `jina-embeddings-v3`. On top of the backbone, `jina-embeddings-v3` has been trained with 5 task-specific adapters for different embedding uses.

**Task-Specific Adapters:**

Include `task` in your request to optimize your downstream application:

+ **retrieval.query**: Used to encode user queries or questions in retrieval tasks.
+ **retrieval.passage**: Used to encode large documents in retrieval tasks at indexing time.
+ **classification**: Used to encode text for text classification tasks.
+ **text-matching**: Used to encode text for similarity matching, such as measuring similarity between two sentences.
+ **separation**: Used for clustering or reranking tasks.


**Matryoshka Representation Learning**:

`jina-embeddings-v3` supports Matryoshka Representation Learning, allowing users to control the embedding dimension with minimal performance loss.  
Include `dimensions` in your request to select the desired dimension.  
By default, **dimensions** is set to 1024, and a number between 256 and 1024 is recommended.  
You can reference the table below for hints on dimension vs. performance:


|         Dimension          | 32 |  64  | 128 |  256   |  512   |   768 |  1024   | 
|:----------------------:|:---------:|:---------:|:-----------:|:---------:|:----------:|:---------:|:---------:|
|  Average Retrieval Performance (nDCG@10)   |   52.54     | 58.54 |    61.64    | 62.72 | 63.16  | 63.3  |   63.35    | 

**Late Chunking in Long-Context Embedding Models**

`jina-embeddings-v3` supports [Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/), the technique to leverage the model's long-context capabilities for generating contextual chunk embeddings. Include `late_chunking=True` in your request to enable contextual chunked representation. When set to true, Jina AI API will concatenate all sentences in the input field and feed them as a single string to the model. Internally, the model embeds this long concatenated string and then performs late chunking, returning a list of embeddings that matches the size of the input list.
"""
logger.info("## Embed text and queries with Jina embedding models through JinaAI API")


text_embed_model = JinaEmbedding(
    api_key=jinaai_api_key,
    model="jina-embeddings-v3",
    task="retrieval.passage",
)

embeddings = text_embed_model.get_text_embedding("This is the text to embed")
logger.debug("Text dim:", len(embeddings))
logger.debug("Text embed:", embeddings[:5])

query_embed_model = JinaEmbedding(
    api_key=jinaai_api_key,
    model="jina-embeddings-v3",
    task="retrieval.query",
    dimensions=512,
)

embeddings = query_embed_model.get_query_embedding(
    "This is the query to embed"
)
logger.debug("Query dim:", len(embeddings))
logger.debug("Query embed:", embeddings[:5])

"""
## Embed images and queries with Jina CLIP through JinaAI API

You can also encode your images and your queries using the JinaEmbedding class
"""
logger.info("## Embed images and queries with Jina CLIP through JinaAI API")


embed_model = JinaEmbedding(
    api_key=jinaai_api_key,
    model="jina-clip-v1",
)

image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcStMP8S3VbNCqOQd7QQQcbvC_FLa1HlftCiJw&s"
im = Image.open(requests.get(image_url, stream=True).raw)
logger.debug("Image:")
display(im)

image_embeddings = embed_model.get_image_embedding(image_url)
logger.debug("Image dim:", len(image_embeddings))
logger.debug("Image embed:", image_embeddings[:5])

text_embeddings = embed_model.get_text_embedding(
    "Logo of a pink blue llama on dark background"
)
logger.debug("Text dim:", len(text_embeddings))
logger.debug("Text embed:", text_embeddings[:5])

cos_sim = dot(image_embeddings, text_embeddings) / (
    norm(image_embeddings) * norm(text_embeddings)
)
logger.debug("Cosine similarity:", cos_sim)

"""
## Embed in batches

You can also embed text in batches, the batch size can be controlled by setting the `embed_batch_size` parameter (the default value will be 10 if not passed, and it should not be larger than 2048)
"""
logger.info("## Embed in batches")

embed_model = JinaEmbedding(
    api_key=jinaai_api_key,
    model="jina-embeddings-v3",
    embed_batch_size=16,
    task="retrieval.passage",
)

embeddings = embed_model.get_text_embedding_batch(
    ["This is the text to embed", "More text can be provided in a batch"]
)

logger.debug(len(embeddings))
logger.debug(embeddings[0][:5])

"""
## Let's build a RAG pipeline using Jina AI Embeddings

#### Download Data
"""
logger.info("## Let's build a RAG pipeline using Jina AI Embeddings")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Imports
"""
logger.info("#### Imports")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))




"""
#### Load Data
"""
logger.info("#### Load Data")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
#### Build index
"""
logger.info("#### Build index")

your_openai_key = "YOUR_OPENAI_KEY"
llm = OllamaFunctionCallingAdapter(api_key=your_openai_key)
embed_model = JinaEmbedding(
    api_key=jinaai_api_key,
    model="jina-embeddings-v3",
    embed_batch_size=16,
    task="retrieval.passage",
)

index = VectorStoreIndex.from_documents(
    documents=documents, embed_model=embed_model
)

"""
#### Build retriever
"""
logger.info("#### Build retriever")

search_query_retriever = index.as_retriever()

search_query_retrieved_nodes = search_query_retriever.retrieve(
    "What happened after the thesis?"
)

for n in search_query_retrieved_nodes:
    display_source_node(n, source_length=2000)

logger.info("\n\n[DONE]", bright=True)
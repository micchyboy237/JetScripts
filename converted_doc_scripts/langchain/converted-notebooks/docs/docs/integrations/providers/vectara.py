from jet.logger import logger
from langchain_vectara import Vectara
from langchain_vectara.vectorstores import (
ChainReranker,
CorpusConfig,
CustomerSpecificReranker,
File,
GenerationConfig,
MmrReranker,
SearchConfig,
VectaraQueryConfig,
)
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
# Vectara

[Vectara](https://vectara.com/) is the trusted AI Assistant and Agent platform which focuses on enterprise readiness for mission-critical applications.
Vectara serverless RAG-as-a-service provides all the components of RAG behind an easy-to-use API, including:
1. A way to extract text from files (PDF, PPT, DOCX, etc)
2. ML-based chunking that provides state of the art performance.
3. The [Boomerang](https://vectara.com/how-boomerang-takes-retrieval-augmented-generation-to-the-next-level-via-grounded-generation/) embeddings model.
4. Its own internal vector database where text chunks and embedding vectors are stored.
5. A query service that automatically encodes the query into embedding, and retrieves the most relevant text segments, including support for [Hybrid Search](https://docs.vectara.com/docs/api-reference/search-apis/lexical-matching) as well as multiple reranking options such as the [multi-lingual relevance reranker](https://www.vectara.com/blog/deep-dive-into-vectara-multilingual-reranker-v1-state-of-the-art-reranker-across-100-languages), [MMR](https://vectara.com/get-diverse-results-and-comprehensive-summaries-with-vectaras-mmr-reranker/), [UDF reranker](https://www.vectara.com/blog/rag-with-user-defined-functions-based-reranking). 
6. An LLM for creating a [generative summary](https://docs.vectara.com/docs/learn/grounded-generation/grounded-generation-overview), based on the retrieved documents (context), including citations.

For more information:
- [Documentation](https://docs.vectara.com/docs/)
- [API Playground](https://docs.vectara.com/docs/rest-api/)
- [Quickstart](https://docs.vectara.com/docs/quickstart)

This notebook shows how to use the basic retrieval functionality, when utilizing Vectara just as a Vector Store (without summarization), incuding: `similarity_search` and `similarity_search_with_score` as well as using the LangChain `as_retriever` functionality.


## Setup

To use the `VectaraVectorStore` you first need to install the partner package.
"""
logger.info("# Vectara")

# !uv pip install -U pip && uv pip install -qU langchain-vectara

"""
# Getting Started

To get started, use the following steps:
1. If you don't already have one, [Sign up](https://www.vectara.com/integrations/langchain) for your free Vectara trial.
2. Within your account you can create one or more corpora. Each corpus represents an area that stores text data upon ingest from input documents. To create a corpus, use the **"Create Corpus"** button. You then provide a name to your corpus as well as a description. Optionally you can define filtering attributes and apply some advanced options. If you click on your created corpus, you can see its name and corpus ID right on the top.
3. Next you'll need to create API keys to access the corpus. Click on the **"Access Control"** tab in the corpus view and then the **"Create API Key"** button. Give your key a name, and choose whether you want query-only or query+index for your key. Click "Create" and you now have an active API key. Keep this key confidential. 

To use LangChain with Vectara, you'll need to have these two values: `corpus_key` and `api_key`.
You can provide `VECTARA_API_KEY` to LangChain in two ways:

1. Include in your environment these two variables: `VECTARA_API_KEY`.

#    For example, you can set these variables using os.environ and getpass as follows:

```python
# import getpass

# os.environ["VECTARA_API_KEY"] = getpass.getpass("Vectara API Key:")
```

2. Add them to the `Vectara` vectorstore constructor:

```python
vectara = Vectara(
    vectara_api_key=vectara_api_key
)
```

In this notebook we assume they are provided in the environment.
"""
logger.info("# Getting Started")


os.environ["VECTARA_API_KEY"] = "<VECTARA_API_KEY>"
os.environ["VECTARA_CORPUS_KEY"] = "VECTARA_CORPUS_KEY"


vectara = Vectara(vectara_api_key=os.getenv("VECTARA_API_KEY"))

"""
First we load the state-of-the-union text into Vectara.

Note that we use the add_files interface which does not require any local processing or chunking - Vectara receives the file content and performs all the necessary pre-processing, chunking and embedding of the file into its knowledge store.

In this case it uses a .txt file but the same works for many other [file types](https://docs.vectara.com/docs/api-reference/indexing-apis/file-upload/file-upload-filetypes).
"""
logger.info("First we load the state-of-the-union text into Vectara.")

corpus_key = os.getenv("VECTARA_CORPUS_KEY")
file_obj = File(
    file_path="../document_loaders/example_data/state_of_the_union.txt",
    metadata={"source": "text_file"},
)
vectara.add_files([file_obj], corpus_key)

"""
## Vectara RAG (retrieval augmented generation)

We now create a `VectaraQueryConfig` object to control the retrieval and summarization options:
* We enable summarization, specifying we would like the LLM to pick the top 7 matching chunks and respond in English

Using this configuration, let's create a LangChain `Runnable` object that encpasulates the full Vectara RAG pipeline, using the `as_rag` method:
"""
logger.info("## Vectara RAG (retrieval augmented generation)")

generation_config = GenerationConfig(
    max_used_search_results=7,
    response_language="eng",
    generation_preset_name="vectara-summary-ext-24-05-med-omni",
    enable_factual_consistency_score=True,
)
search_config = SearchConfig(
    corpora=[CorpusConfig(corpus_key=corpus_key)],
    limit=25,
    reranker=ChainReranker(
        rerankers=[
            CustomerSpecificReranker(reranker_id="rnk_272725719", limit=100),
            MmrReranker(diversity_bias=0.2, limit=100),
        ]
    ),
)

config = VectaraQueryConfig(
    search=search_config,
    generation=generation_config,
)

query_str = "what did Biden say?"

rag = vectara.as_rag(config)
rag.invoke(query_str)["answer"]

"""
We can also use the streaming interface like this:
"""
logger.info("We can also use the streaming interface like this:")

output = {}
curr_key = None
for chunk in rag.stream(query_str):
    for key in chunk:
        if key not in output:
            output[key] = chunk[key]
        else:
            output[key] += chunk[key]
        if key == "answer":
            logger.debug(chunk[key], end="", flush=True)
        curr_key = key

"""
For more details about Vectara as VectorStore [go to this notebook](../vectorstores/vectara.ipynb).

## Vectara Chat

In most uses of LangChain to create chatbots, one must integrate a special `memory` component that maintains the history of chat sessions and then uses that history to ensure the chatbot is aware of conversation history.

With Vectara Chat - all of that is performed in the backend by Vectara automatically.
"""
logger.info("## Vectara Chat")

generation_config = GenerationConfig(
    max_used_search_results=7,
    response_language="eng",
    generation_preset_name="vectara-summary-ext-24-05-med-omni",
    enable_factual_consistency_score=True,
)
search_config = SearchConfig(
    corpora=[CorpusConfig(corpus_key=corpus_key, limit=25)],
    reranker=MmrReranker(diversity_bias=0.2),
)

config = VectaraQueryConfig(
    search=search_config,
    generation=generation_config,
)


bot = vectara.as_chat(config)

bot.invoke("What did the president say about Ketanji Brown Jackson?")["answer"]

"""
For more details about Vectara chat [go to this notebook](../chat/vectara.ipynb).

## Vectara as self-querying retriever
Vectara offers Intelligent Query Rewriting option which  enhances search precision by automatically generating metadata filter expressions from natural language queries. This capability analyzes user queries, extracts relevant metadata filters, and rephrases the query to focus on the core information need. For more details [go to this notebook](../retrievers/self_query/vectara_self_query.ipynb).

## Vectara tools
Vectara provides serval tools that can be used with Langchain. For more details [go to this notebook](../tools/vectara.ipynb)
"""
logger.info("## Vectara as self-querying retriever")


logger.info("\n\n[DONE]", bright=True)
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.packs.koda_retriever import KodaRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Alpha Tuning w/ Koda Retriever

*For this example non-production ready infrastructure is leveraged*
More specifically, the default sample data provided in a free start instance of [Pinecone](https://www.pinecone.io/) is used. This data consists of movie scripts and their summaries embedded in a free Pinecone vector database.

### Agenda
- Fixture Setup
- Alpha Tuning Setup
- Koda Retriever: Retrieval 

A quick overview/visual on how alpha tuning works: (excuse the weird colors, my color settings on Windows was weirdly interacting w/ Google Sheets and made some colors useless)
![alpha-tuning](https://i.imgur.com/zxCXqGb.png)

### Example Context
Let's say you're building a query engine or agent that is expected to answer questions on Deep Learning, AI, RAG Architecture, and adjacent topics. As a result of this, your team has narrowed down three main query classifications and associated alpha values with those classifications. Your alpha values were determined by incrementally decreasing the alpha value from 1 to 0 and for each new alpha value, several queries are run and evaluated. Repeating this process for each category should yield an optimal alpha value for each category over your data. This process should be repeated periodically as your data expands or changes. 

With the categories and corresponding alpha values uncovered, these are our categories:
- Concept Seeking Queries *(α: 0.2)*
- Fact Seeking Queries *(α: .6)*
- Queries w/ Misspellings *(α: 1)*

Clearly, these categories have very different biases towards one end of the retrieval spectrum. The default for Llama Index hybrid retrievers is 0.5.
"""
logger.info("# Alpha Tuning w/ Koda Retriever")


"""
## Setup

Building *required objects* for a Koda Retriever.
- Vector Index
- LLM/Model

Other objects are *optional*, and will be used if provided:
- Reranker
- Custom categories & corresponding alpha weights
- A custom model trained on the custom info above
"""
logger.info("## Setup")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("sample-movies")

Settings.llm = MLX()
Settings.embed_model = MLXEmbedding()

vector_store = PineconeVectorStore(pinecone_index=index, text_key="summary")
vector_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=Settings.embed_model
)

reranker = LLMRerank(llm=Settings.llm)  # optional

"""
## Defining Categories & Alpha Values

We need to first input our categories and alpha values in a format that Koda can understand.

### Important Considerations:
If you provide these custom categories and no custom model, these values will be input as few-shot context training for whatever model is provided to Koda Retriever. Otherwise, if a custom model is provided and has been trained on the data that would otherwise be provided below, ensure the keys of the categories dictionary matches the expected labels of the custom model. Likewise, do NOT provide any examples or a description.
"""
logger.info("## Defining Categories & Alpha Values")

categories = {  # key, #alpha, description [and examples]
    "concept seeking query": {
        "alpha": 0.2,
        "description": "Abstract questions, usually on a specific topic, that require multiple sentences to answer",
        "examples": [
            "What is the dual-encoder architecture used in recent works on dense retrievers?",
            "Why should I use semantic search to rank results?",
        ],
    },
    "fact seeking query": {
        "alpha": 0.6,
        "description": "Queries with a single, clear answer",
        "examples": [
            "What is the total number of propositions the English Wikipedia dump is segmented into in FACTOID WIKI?",
            "How many documents are semantically ranked?",
        ],
    },
    "queries with misspellings": {
        "alpha": 1,
        "description": "Queries with typos, transpositions and common misspellings introduced",
        "examples": [
            "What is the advntage of prposition retrieval over sentnce or passage retrieval?",
            "Ho w mny documents are samantically r4nked",
        ],
    },
}

"""
## Building Koda Retriever
"""
logger.info("## Building Koda Retriever")

retriever = KodaRetriever(
    index=vector_index,
    llm=Settings.llm,
    matrix=categories,  # koda now knows to use these categories
    reranker=reranker,  # optional
    verbose=True,
)

"""
## Retrieving w/ Koda Retriever
"""
logger.info("## Retrieving w/ Koda Retriever")

query = "Can you explain the Jurassic Park as a business as it was supposed to operate inside the movie's lore or timeline?"
results = retriever.retrieve(query)

results

"""
Those results don't look quite palletteable though. For that, lets look into making the response more *natural*. For that we'll likely need a Query Engine.

# Query Engine w/ Koda Retriever

Query Engines are [Llama Index abstractions](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/root.html) that combine retrieval and synthesization of an LLM to interpret the results given by a retriever into a natural language response to the original query. They are themselves an end-to-end pipeline from query to natural langauge response.
"""
logger.info("# Query Engine w/ Koda Retriever")

query_engine = RetrieverQueryEngine.from_args(retriever=retriever)

response = query_engine.query(query)

str(response)

logger.info("\n\n[DONE]", bright=True)
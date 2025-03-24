"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/citation_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# CitationQueryEngine

This notebook walks through how to use the CitationQueryEngine

The CitationQueryEngine can be used with any existing index.
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-embeddings-ollama
# %pip install llama-index-llms-ollama

# !pip install llama-index

"""
## Setup
"""


from llama_index.embeddings.ollama import OllamaEmbedding
from jet.llm.ollama.base import Ollama
from llama_index.core import Settings
Settings.llm = Ollama(
    model="llama3.2", request_timeout=300.0, context_window=4096)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

"""
## Download Data
"""

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

if not os.path.exists("./citation"):
    documents = SimpleDirectoryReader(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
    index = VectorStoreIndex.from_documents(
        documents,
    )
    index.storage_context.persist(persist_dir="./citation")
else:
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./citation"),
    )

"""
## Create the CitationQueryEngine w/ Default Arguments
"""

query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    citation_chunk_size=512,
)

response = query_engine.query("What did the author do growing up?")

print(response)

print(len(response.source_nodes))

"""
### Inspecting the Actual Source
Sources start counting at 1, but python arrays start counting at zero!

Let's confirm the source makes sense.
"""

print(response.source_nodes[0].node.get_text())

print(response.source_nodes[1].node.get_text())

"""
## Adjusting Settings

Note that setting the chunk size larger than the original chunk size of the nodes will have no effect.

The default node chunk size is 1024, so here, we are not making our citation nodes any more granular.
"""

query_engine = CitationQueryEngine.from_args(
    index,
    citation_chunk_size=1024,
    similarity_top_k=3,
)

response = query_engine.query("What did the author do growing up?")

print(response)

print(len(response.source_nodes))

"""
### Inspecting the Actual Source
Sources start counting at 1, but python arrays start counting at zero!

Let's confirm the source makes sense.
"""

print(response.source_nodes[0].node.get_text())

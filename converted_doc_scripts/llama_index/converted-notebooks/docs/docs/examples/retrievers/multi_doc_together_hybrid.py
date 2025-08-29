from jet.models.config import MODELS_CACHE_DIR
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core import Document
from llama_index.core import QueryBundle
from llama_index.core import SummaryIndex
from llama_index.core.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.readers.file import UnstructuredReader
from pathlib import Path
from tqdm.notebook import tqdm
from typing import List, Any, Optional
import os
import pickle
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/together.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Chunk + Document Hybrid Retrieval with Long-Context Embeddings (Together.ai) 

This notebook shows how to use long-context together.ai embedding models for advanced RAG. We index each document by running the embedding model over the entire document text, as well as embedding each chunk. We then define a custom retriever that can compute both node similarity as well as document similarity.

Visit https://together.ai and sign up to get an API key.

## Setup and Download Data

We load in our documentation. For the sake of speed we load in just 10 pages, but of course if you want to stress test your model you should load in all of it.
"""
logger.info(
    "# Chunk + Document Hybrid Retrieval with Long-Context Embeddings (Together.ai)")

# %pip install llama-index-embeddings-together
# %pip install llama-index-llms-ollama
# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-readers-file

domain = "docs.llamaindex.ai"
docs_url = "https://docs.llamaindex.ai/en/latest/"
# !wget -e robots=off --recursive --no-clobber --page-requisites --html-extension --convert-links --restrict-file-names=windows --domains {domain} --no-parent {docs_url}


reader = UnstructuredReader()

all_html_files = [
    "docs.llamaindex.ai/en/latest/index.html",
    "docs.llamaindex.ai/en/latest/contributing/contributing.html",
    "docs.llamaindex.ai/en/latest/understanding/understanding.html",
    "docs.llamaindex.ai/en/latest/understanding/using_llms/using_llms.html",
    "docs.llamaindex.ai/en/latest/understanding/using_llms/privacy.html",
    "docs.llamaindex.ai/en/latest/understanding/loading/llamahub.html",
    "docs.llamaindex.ai/en/latest/optimizing/production_rag.html",
    "docs.llamaindex.ai/en/latest/module_guides/models/llms.html",
]


doc_limit = 10

docs = []
for idx, f in enumerate(all_html_files):
    if idx > doc_limit:
        break
    logger.debug(f"Idx {idx}/{len(all_html_files)}")
    loaded_docs = reader.load_data(file=f, split_documents=True)
    start_idx = 64
    loaded_doc = Document(
        id_=str(f),
        text="\n\n".join([d.get_content() for d in loaded_docs[start_idx:]]),
        metadata={"path": str(f)},
    )
    logger.debug(str(f))
    docs.append(loaded_doc)

"""
## Building Hybrid Retrieval with Chunk Embedding + Parent Embedding

Define a custom retriever that does the following:
- First retrieve relevant chunks based on embedding similarity
- For each chunk, lookup the source document embedding.
- Weight it by an alpha.

This is essentially vector retrieval with a reranking step that reweights the node similarities.
"""
logger.info(
    "## Building Hybrid Retrieval with Chunk Embedding + Parent Embedding")


api_key = "<api_key>"

embed_model = TogetherEmbedding(
    model_name="togethercomputer/m2-bert-80M-32k-retrieval", api_key=api_key
)

llm = OllamaFunctionCallingAdapter(temperature=0, model="llama3.2")

"""
### Create Document Store 

Create docstore for original documents. Embed each document, and put in docstore.

We will refer to this later in our hybrid retrieval algorithm!
"""
logger.info("### Create Document Store")


for doc in docs:
    embedding = embed_model.get_text_embedding(doc.get_content())
    doc.embedding = embedding

docstore = SimpleDocumentStore()
docstore.add_documents(docs)

"""
### Build Vector Index

Let's build the vector index of chunks. Each chunk will also have a reference to its source document through its `index_id` (which can then be used to lookup the source document in the docstore).
"""
logger.info("### Build Vector Index")


def build_index(docs, out_path: str = "storage/chunk_index"):
    nodes = []

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=70)
    for idx, doc in enumerate(tqdm(docs)):

        cur_nodes = splitter.get_nodes_from_documents([doc])
        for cur_node in cur_nodes:
            file_path = doc.metadata["path"]
            new_node = IndexNode(
                text=cur_node.text or "None",
                index_id=str(file_path),
                metadata=doc.metadata
            )
            nodes.append(new_node)
    logger.debug("num nodes: " + str(len(nodes)))

    if not os.path.exists(out_path):
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        index.set_index_id("simple_index")
        index.storage_context.persist(f"./{out_path}")
    else:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./{out_path}"
        )
        index = load_index_from_storage(
            storage_context, index_id="simple_index", embed_model=embed_model
        )

    return index


index = build_index(docs)

"""
### Define Hybrid Retriever

We define a hybrid retriever that can first fetch chunks by vector similarity, and then reweight it based on similarity with the parent document (using an alpha parameter).
"""
logger.info("### Define Hybrid Retriever")


class HybridRetriever(BaseRetriever):
    """Hybrid retriever."""

    def __init__(
        self,
        vector_index,
        docstore,
        similarity_top_k: int = 2,
        out_top_k: Optional[int] = None,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(**kwargs)
        self._vector_index = vector_index
        self._embed_model = vector_index._embed_model
        self._retriever = vector_index.as_retriever(
            similarity_top_k=similarity_top_k
        )
        self._out_top_k = out_top_k or similarity_top_k
        self._docstore = docstore
        self._alpha = alpha

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        nodes = self._retriever.retrieve(query_bundle.query_str)

        docs = [self._docstore.get_document(n.node.index_id) for n in nodes]
        doc_embeddings = [d.embedding for d in docs]
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )

        doc_similarities, doc_idxs = get_top_k_embeddings(
            query_embedding, doc_embeddings
        )

        result_tups = []
        for doc_idx, doc_similarity in zip(doc_idxs, doc_similarities):
            node = nodes[doc_idx]
            full_similarity = (self._alpha * node.score) + (
                (1 - self._alpha) * doc_similarity
            )
            logger.debug(
                f"Doc {doc_idx} (node score, doc similarity, full similarity): {(node.score, doc_similarity, full_similarity)}"
            )
            result_tups.append((full_similarity, node))

        result_tups = sorted(result_tups, key=lambda x: x[0], reverse=True)
        for full_score, node in result_tups:
            node.score = full_score

        return [n for _, n in result_tups][:out_top_k]


top_k = 10
out_top_k = 3
hybrid_retriever = HybridRetriever(
    index, docstore, similarity_top_k=top_k, out_top_k=3, alpha=0.5
)
base_retriever = index.as_retriever(similarity_top_k=out_top_k)


def show_nodes(nodes, out_len: int = 200):
    for idx, n in enumerate(nodes):
        logger.debug(f"\n\n >>>>>>>>>>>> ID {n.id_}: {n.metadata['path']}")
        logger.debug(n.get_content()[:out_len])


query_str = "Tell me more about the LLM interface and where they're used"

nodes = hybrid_retriever.retrieve(query_str)

show_nodes(nodes)

base_nodes = base_retriever.retrieve(query_str)

show_nodes(base_nodes)

"""
## Run Some Queries
"""
logger.info("## Run Some Queries")


query_engine = RetrieverQueryEngine(hybrid_retriever)
base_query_engine = index.as_query_engine(similarity_top_k=out_top_k)

response = query_engine.query(query_str)
logger.debug(str(response))

base_response = base_query_engine.query(query_str)
logger.debug(str(base_response))

logger.info("\n\n[DONE]", bright=True)

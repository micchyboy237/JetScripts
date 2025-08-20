from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import (
PrevNextNodePostprocessor,
AutoPrevNextNodePostprocessor,
)
from llama_index.core.settings import Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/PrevNextPostprocessorDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Forward/Backward Augmentation

Showcase capabilities of leveraging Node relationships on top of PG's essay

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Forward/Backward Augmentation")

# !pip install llama-index


"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
### Parse Documents into Nodes, add to Docstore
"""
logger.info("### Parse Documents into Nodes, add to Docstore")



documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


Settings.chunk_size = 512

nodes = Settings.node_parser.get_nodes_from_documents(documents)

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

"""
### Build Index
"""
logger.info("### Build Index")

index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
### Add PrevNext Node Postprocessor
"""
logger.info("### Add PrevNext Node Postprocessor")

node_postprocessor = PrevNextNodePostprocessor(docstore=docstore, num_nodes=4)

query_engine = index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[node_postprocessor],
    response_mode="tree_summarize",
)
response = query_engine.query(
    "What did the author do after handing off Y Combinator to Sam Altman?",
)

logger.debug(response)

query_engine = index.as_query_engine(
    similarity_top_k=1, response_mode="tree_summarize"
)
response = query_engine.query(
    "What did the author do after handing off Y Combinator to Sam Altman?",
)

logger.debug(response)

query_engine = index.as_query_engine(
    similarity_top_k=3, response_mode="tree_summarize"
)
response = query_engine.query(
    "What did the author do after handing off Y Combinator to Sam Altman?",
)

logger.debug(response)

"""
### Add Auto Prev/Next Node Postprocessor
"""
logger.info("### Add Auto Prev/Next Node Postprocessor")

node_postprocessor = AutoPrevNextNodePostprocessor(
    docstore=docstore,
    num_nodes=3,
    verbose=True,
)

query_engine = index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[node_postprocessor],
    response_mode="tree_summarize",
)
response = query_engine.query(
    "What did the author do after handing off Y Combinator to Sam Altman?",
)

logger.debug(response)

response = query_engine.query(
    "What did the author do during his time at Y Combinator?",
)

logger.debug(response)

response = query_engine.query(
    "What did the author do before handing off Y Combinator to Sam Altman?",
)

logger.debug(response)

response = query_engine.query(
    "What did the author do before handing off Y Combinator to Sam Altman?",
)

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
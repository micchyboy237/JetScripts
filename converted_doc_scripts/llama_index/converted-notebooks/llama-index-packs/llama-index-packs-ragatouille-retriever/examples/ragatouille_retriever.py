from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.packs.ragatouille_retriever import RAGatouilleRetrieverPack
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
# RAGatouille Retriever Llama Pack 

RAGatouille is a [cool library](https://github.com/bclavie/RAGatouille) that lets you use e.g. ColBERT and other SOTA retrieval models in your RAG pipeline. You can use it to either run inference on ColBERT, or use it to train/fine-tune models.

This LlamaPack shows you an easy way to bundle RAGatouille into your RAG pipeline. We use RAGatouille to index a corpus of documents (by default using colbertv2.0), and then we combine it with LlamaIndex query modules to synthesize an answer with an LLM.
"""
logger.info("# RAGatouille Retriever Llama Pack")

# %pip install llama-index-llms-ollama
# %pip install llama-index-packs-ragatouille-retriever



"""
## Load Documents

Here we load the ColBERTv2 paper: https://arxiv.org/pdf/2112.01488.pdf.
"""
logger.info("## Load Documents")

# !wget "https://arxiv.org/pdf/2004.12832.pdf" -O colbertv1.pdf


reader = SimpleDirectoryReader(input_files=["colbertv1.pdf"])
docs = reader.load_data()

"""
## Create Pack
"""
logger.info("## Create Pack")

index_name = "my_index"
ragatouille_pack = RAGatouilleRetrieverPack(
    docs, llm=MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats"), index_name=index_name, top_k=5
)

"""
## Try out Pack

We try out both the individual modules in the pack as well as running it e2e!
"""
logger.info("## Try out Pack")


retriever = ragatouille_pack.get_modules()["retriever"]
nodes = retriever.retrieve("How does ColBERTv2 compare with other BERT models?")

for node in nodes:
    display_source_node(node)

RAG = ragatouille_pack.get_modules()["RAG"]
results = RAG.search(
    "How does ColBERTv2 compare with other BERT models?", index_name=index_name, k=4
)
results

response = ragatouille_pack.run("How does ColBERTv2 compare with other BERT models?")
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
from self_rag_pack.base import SelfRAGQueryEngine
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
# Simple Self RAG Notebook

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-self-rag/examples/self_rag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This LlamaPack implements short form the [self-RAG paper by Akari et al.](https://arxiv.org/pdf/2310.11511.pdf).

Novel framework called Self-Reflective Retrieval-Augmented Generation (SELF-RAG). Which aims to enhance the quality and factuality of large language models (LLMs) by combining retrieval and self-reflection mechanisms.

The implementation is adapted from the author [implementation](https://github.com/AkariAsai/self-rag)
A full notebook guide can be found [here](https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/self_rag/self_rag.ipynb).

## Setup
"""
logger.info("# Simple Self RAG Notebook")


documents = [
    Document(
        text="A group of penguins, known as a 'waddle' on land, shuffled across the Antarctic ice, their tuxedo-like plumage standing out against the snow."
    ),
    Document(
        text="Emperor penguins, the tallest of all penguin species, can dive deeper than any other bird, reaching depths of over 500 meters."
    ),
    Document(
        text="Penguins' black and white coloring is a form of camouflage called countershading; from above, their black back blends with the ocean depths, and from below, their white belly matches the bright surface."
    ),
    Document(
        text="Despite their upright stance, penguins are birds that cannot fly; their wings have evolved into flippers, making them expert swimmers."
    ),
    Document(
        text="The fastest species, the Gentoo penguin, can swim up to 36 kilometers per hour, using their flippers and streamlined bodies to slice through the water."
    ),
    Document(
        text="Penguins are social birds; many species form large colonies for breeding, which can number in the tens of thousands."
    ),
    Document(
        text="Intriguingly, penguins have excellent hearing and rely on distinct calls to identify their mates and chicks amidst the noisy colonies."
    ),
    Document(
        text="The smallest penguin species, the Little Blue Penguin, stands just about 40 cm tall and is found along the coastlines of southern Australia and New Zealand."
    ),
    Document(
        text="During the breeding season, male Emperor penguins endure the harsh Antarctic winter for months, fasting and incubating their eggs, while females hunt at sea."
    ),
    Document(
        text="Penguins consume a variety of seafood; their diet mainly consists of fish, squid, and krill, which they catch on their diving expeditions."
    ),
]

index = VectorStoreIndex.from_documents(documents)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

"""
## Load Pack / Setup

Now we do `download_llama_pack` to load the Self-RAG LlamaPack (you can also import the module directly if using the llama-hub package).

We will also optionally setup observability/tracing so we can observe the intermediate steps.
"""
logger.info("## Load Pack / Setup")


download_llama_pack(
    "SelfRAGPack",
    "./self_rag_pack",
    skip_load=True,
)

download_dir = "/home/mmaatouk/tmp"  # Replace
# !pip3 install -q huggingface-hub
# !huggingface-cli download m4r1/selfrag_llama2_7b-GGUF selfrag_llama2_7b.q4_k_m.gguf --local-dir {download_dir} --local-dir-use-symlinks False


model_path = Path(download_dir) / "selfrag_llama2_7b.q4_k_m.gguf"
query_engine = SelfRAGQueryEngine(str(model_path), retriever, verbose=True)

"""
## Try out some Queries

Now let's try out our `SelfRAGQueryEngine`!
"""
logger.info("## Try out some Queries")

response = query_engine.query("Which genre the book pride and prejudice?")

response = query_engine.query("How tall is the smallest penguins?")

logger.info("\n\n[DONE]", bright=True)
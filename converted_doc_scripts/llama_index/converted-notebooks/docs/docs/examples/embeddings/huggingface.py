from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/embeddings/huggingface.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Local Embeddings with HuggingFace

LlamaIndex has support for HuggingFace embedding models, including Sentence Transformer models like BGE, Mixedbread, Nomic, Jina, E5, etc. We can use these models to create embeddings for our documents and queries for retrieval.

Furthermore, we provide utilities to create and use ONNX and OpenVINO models using the [Optimum library](https://huggingface.co/docs/optimum) from HuggingFace.

## HuggingFaceEmbedding

The base `HuggingFaceEmbedding` class is a generic wrapper around any HuggingFace model for embeddings. All [embedding models](https://huggingface.co/models?library=sentence-transformers) on Hugging Face should work. You can refer to the [embeddings leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for more recommendations.

This class depends on the sentence-transformers package, which you can install with `pip install sentence-transformers`.

NOTE: if you were previously using a `HuggingFaceEmbeddings` from LangChain, this should give equivalent results.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Local Embeddings with HuggingFace")

# %pip install llama-index-embeddings-huggingface

# !pip install llama-index


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

embeddings = embed_model.get_text_embedding("Hello World!")
logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
## Benchmarking

Let's try comparing using a classic large document -- the IPCC climate report, chapter 3.
"""
logger.info("## Benchmarking")

# !curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf


documents = SimpleDirectoryReader(
    input_files=["IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

"""
### Base HuggingFace Embeddings
"""
logger.info("### Base HuggingFace Embeddings")


embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
    embed_batch_size=8,
)
test_embeds = embed_model.get_text_embedding("Hello World!")

Settings.embed_model = embed_model

# %%timeit -r 1 -n 1
index = VectorStoreIndex.from_documents(documents, show_progress=True)

"""
### ONNX Embeddings
"""
logger.info("### ONNX Embeddings")

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
    backend="onnx",
    model_kwargs={
        "provider": "CPUExecutionProvider"
    },  # For ONNX, you can specify the provider, see https://sbert.net/docs/sentence_transformer/usage/efficiency.html
)
test_embeds = embed_model.get_text_embedding("Hello World!")

Settings.embed_model = embed_model

# %%timeit -r 1 -n 1
index = VectorStoreIndex.from_documents(documents, show_progress=True)

"""
### OpenVINO Embeddings
"""
logger.info("### OpenVINO Embeddings")

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu",
    backend="openvino",  # OpenVINO is very strong on CPUs
    revision="refs/pr/16",  # BAAI/bge-small-en-v1.5 itself doesn't have an OpenVINO model currently, but there's a PR with it that we can load: https://huggingface.co/BAAI/bge-small-en-v1.5/discussions/16
    model_kwargs={
        "file_name": "openvino_model_qint8_quantized.xml"
    },  # If we're using an optimized/quantized model, we need to specify the file name like this
)
test_embeds = embed_model.get_text_embedding("Hello World!")

Settings.embed_model = embed_model

# %%timeit -r 1 -n 1
index = VectorStoreIndex.from_documents(documents, show_progress=True)

"""
### References

* [Local Embedding Models](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/#local-embedding-models) explains more about using local models like these.
* [Sentence Transformers > Speeding up Inference](https://sbert.net/docs/sentence_transformer/usage/efficiency.html) contains extensive documentation on how to use the backend options effectively, including optimization and quantization for ONNX and OpenVINO.
"""
logger.info("### References")

logger.info("\n\n[DONE]", bright=True)
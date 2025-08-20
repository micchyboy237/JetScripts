from PIL import Image
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_openvino import (
OpenVINOClipEmbedding,
)
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.embeddings.openvino_genai import OpenVINOGENAIEmbedding
from numpy import dot
from numpy.linalg import norm
import os
import requests
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
# Local Embeddings with OpenVINO

[OpenVINO™](https://github.com/openvinotoolkit/openvino) is an open-source toolkit for optimizing and deploying AI inference. The OpenVINO™ Runtime supports various hardware [devices](https://github.com/openvinotoolkit/openvino?tab=readme-ov-file#supported-hardware-matrix) including x86 and ARM CPUs, and Intel GPUs. It can help to boost deep learning performance in Computer Vision, Automatic Speech Recognition, Natural Language Processing and other common tasks.

Hugging Face embedding model can be supported by OpenVINO through ``OpenVINOEmbedding`` or ``OpenVINOGENAIEmbedding``class, and OpenClip model can be through ``OpenVINOClipEmbedding`` class.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Local Embeddings with OpenVINO")

# %pip install llama-index-embeddings-openvino

# !pip install llama-index

"""
## Model Exporter

It is possible to export your model to the OpenVINO IR format with `create_and_save_openvino_model` function, and load the model from local folder.
"""
logger.info("## Model Exporter")


OpenVINOEmbedding.create_and_save_openvino_model(
    "BAAI/bge-small-en-v1.5", "./bge_ov"
)

"""
## Model Loading
If you have an Intel GPU, you can specify `device="gpu"` to run inference on it.
"""
logger.info("## Model Loading")

ov_embed_model = OpenVINOEmbedding(model_id_or_path="./bge_ov", device="cpu")

embeddings = ov_embed_model.get_text_embedding("Hello World!")
logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
## Model Loading with OpenVINO GenAI

To avoid the dependencies of PyTorch in runtime, you can load your local embedding model with ``OpenVINOGENAIEmbedding``class.
"""
logger.info("## Model Loading with OpenVINO GenAI")

# %pip install llama-index-embeddings-openvino-genai


ov_embed_model = OpenVINOGENAIEmbedding(model_path="./bge_ov", device="CPU")

embeddings = ov_embed_model.get_text_embedding("Hello World!")
logger.debug(len(embeddings))
logger.debug(embeddings[:5])

"""
## OpenClip Model Exporter
Class `OpenVINOClipEmbedding` can support exporting and loading open_clip models with OpenVINO runtime.
"""
logger.info("## OpenClip Model Exporter")

# %pip install open_clip_torch


OpenVINOClipEmbedding.create_and_save_openvino_model(
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "ViT-B-32-ov",
)

"""
## MultiModal Model Loading
If you have an Intel GPU, you can specify `device="GPU"` to run inference on it.
"""
logger.info("## MultiModal Model Loading")

ov_clip_model = OpenVINOClipEmbedding(
    model_id_or_path="./ViT-B-32-ov", device="CPU"
)

"""
## Embed images and queries with OpenVINO
"""
logger.info("## Embed images and queries with OpenVINO")


image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcStMP8S3VbNCqOQd7QQQcbvC_FLa1HlftCiJw&s"
im = Image.open(requests.get(image_url, stream=True).raw)
logger.debug("Image:")
display(im)

im.save("logo.jpg")
image_embeddings = ov_clip_model.get_image_embedding("logo.jpg")
logger.debug("Image dim:", len(image_embeddings))
logger.debug("Image embed:", image_embeddings[:5])

text_embeddings = ov_clip_model.get_text_embedding(
    "Logo of a pink blue llama on dark background"
)
logger.debug("Text dim:", len(text_embeddings))
logger.debug("Text embed:", text_embeddings[:5])

cos_sim = dot(image_embeddings, text_embeddings) / (
    norm(image_embeddings) * norm(text_embeddings)
)
logger.debug("Cosine similarity:", cos_sim)

"""
For more information refer to:

* [OpenVINO LLM guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html).

* [OpenVINO Documentation](https://docs.openvino.ai/2024/home.html).

* [OpenVINO Get Started Guide](https://www.intel.com/content/www/us/en/content-details/819067/openvino-get-started-guide.html).

* [RAG example with LlamaIndex](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-rag-llamaindex).
"""
logger.info("For more information refer to:")

logger.info("\n\n[DONE]", bright=True)
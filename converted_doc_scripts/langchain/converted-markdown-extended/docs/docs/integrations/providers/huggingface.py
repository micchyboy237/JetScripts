from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain_community.agent_toolkits.load_tools import load_huggingface_tool
from langchain_community.document_loaders import HuggingFaceModelLoader
from langchain_community.document_loaders import ImageCaptionLoader
from langchain_community.document_loaders.hugging_face_dataset import HuggingFaceDatasetLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.tools.audio import HuggingFaceTextToSpeechModelInference
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFacePipeline
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
# Hugging Face

All functionality related to [Hugging Face Hub](https://huggingface.co/) and libraries like [transformers](https://huggingface.co/docs/transformers/index), [sentence transformers](https://sbert.net/), and [datasets](https://huggingface.co/docs/datasets/index).

> [Hugging Face](https://huggingface.co/) is an AI platform with all major open source models, datasets, MCPs, and demos.
> It supplies model inference locally and via serverless [Inference Providers](https://huggingface.co/docs/inference-providers).
>
> You can use [Inference Providers](https://huggingface.co/docs/inference-providers) to run open source models like DeepSeek R1 on scalable serverless infrastructure.

## Installation

Most of the Hugging Face integrations are available in the `langchain-huggingface` package.
"""
logger.info("# Hugging Face")

pip install langchain-huggingface

"""
## Chat models

### ChatHuggingFace

We can use the `Hugging Face` LLM classes or directly use the `ChatHuggingFace` class.

See a [usage example](/docs/integrations/chat/huggingface).
"""
logger.info("## Chat models")


"""
## LLMs

### HuggingFaceEndpoint

We can use the `HuggingFaceEndpoint` class to run open source models via serverless [Inference Providers](https://huggingface.co/docs/inference-providers) or via dedicated [Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated).

See a [usage example](/docs/integrations/llms/huggingface_endpoint).
"""
logger.info("## LLMs")


"""
### HuggingFacePipeline

We can use the `HuggingFacePipeline` class to run open source models locally.

See a [usage example](/docs/integrations/llms/huggingface_pipelines).
"""
logger.info("### HuggingFacePipeline")


"""
## Embedding Models

### HuggingFaceEmbeddings

We can use the `HuggingFaceEmbeddings` class to run open source embedding models locally.

See a [usage example](/docs/integrations/text_embedding/huggingfacehub).
"""
logger.info("## Embedding Models")


"""
### HuggingFaceEndpointEmbeddings

We can use the `HuggingFaceEndpointEmbeddings` class to run open source embedding models via a dedicated [Inference Endpoint](https://huggingface.co/inference-endpoints/dedicated).

See a [usage example](/docs/integrations/text_embedding/huggingfacehub).
"""
logger.info("### HuggingFaceEndpointEmbeddings")


"""
### HuggingFaceInferenceAPIEmbeddings

We can use the `HuggingFaceInferenceAPIEmbeddings` class to run open source embedding models via [Inference Providers](https://huggingface.co/docs/inference-providers).

See a [usage example](/docs/integrations/text_embedding/huggingfacehub).
"""
logger.info("### HuggingFaceInferenceAPIEmbeddings")


"""
### HuggingFaceInstructEmbeddings

We can use the `HuggingFaceInstructEmbeddings` class to run open source embedding models locally.

See a [usage example](/docs/integrations/text_embedding/instruct_embeddings).
"""
logger.info("### HuggingFaceInstructEmbeddings")


"""
### HuggingFaceBgeEmbeddings

>[BGE models on the HuggingFace](https://huggingface.co/BAAI/bge-large-en-v1.5) are one of [the best open-source embedding models](https://huggingface.co/spaces/mteb/leaderboard).
>BGE model is created by the [Beijing Academy of Artificial Intelligence (BAAI)](https://en.wikipedia.org/wiki/Beijing_Academy_of_Artificial_Intelligence). `BAAI` is a private non-profit organization engaged in AI research and development.

See a [usage example](/docs/integrations/text_embedding/bge_huggingface).
"""
logger.info("### HuggingFaceBgeEmbeddings")


"""
## Document Loaders

### Hugging Face dataset

>[Hugging Face Hub](https://huggingface.co/docs/hub/index) is home to over 75,000
> [datasets](https://huggingface.co/docs/hub/index#datasets) in more than 100 languages
> that can be used for a broad range of tasks across NLP, Computer Vision, and Audio.
> They used for a diverse range of tasks such as translation, automatic speech
> recognition, and image classification.

We need to install `datasets` python package.
"""
logger.info("## Document Loaders")

pip install datasets

"""
See a [usage example](/docs/integrations/document_loaders/hugging_face_dataset).
"""
logger.info("See a [usage example](/docs/integrations/document_loaders/hugging_face_dataset).")


"""
### Hugging Face model loader

>Load model information from `Hugging Face Hub`, including README content.
>
>This loader interfaces with the `Hugging Face Models API` to fetch
> and load model metadata and README files.
> The API allows you to search and filter models based on
> specific criteria such as model tags, authors, and more.
"""
logger.info("### Hugging Face model loader")


"""
### Image captions

It uses the Hugging Face models to generate image captions.

We need to install several python packages.
"""
logger.info("### Image captions")

pip install transformers pillow

"""
See a [usage example](/docs/integrations/document_loaders/image_captions).
"""
logger.info("See a [usage example](/docs/integrations/document_loaders/image_captions).")


"""
## Tools

### Hugging Face Hub Tools

>[Hugging Face Tools](https://huggingface.co/docs/transformers/v4.29.0/en/custom_tools)
> support text I/O and are loaded using the `load_huggingface_tool` function.

We need to install several python packages.
"""
logger.info("## Tools")

pip install transformers huggingface_hub

"""
See a [usage example](/docs/integrations/tools/huggingface_tools).
"""
logger.info("See a [usage example](/docs/integrations/tools/huggingface_tools).")


"""
### Hugging Face Text-to-Speech Model Inference.

> It is a wrapper around `Ollama Text-to-Speech API`.
"""
logger.info("### Hugging Face Text-to-Speech Model Inference.")


logger.info("\n\n[DONE]", bright=True)
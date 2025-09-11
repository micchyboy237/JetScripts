from jet.logger import logger
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from langchain_community.embeddings import OpenVINOEmbeddings
from pathlib import Path
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
# OpenVINO
[OpenVINO™](https://github.com/openvinotoolkit/openvino) is an open-source toolkit for optimizing and deploying AI inference. The OpenVINO™ Runtime supports various hardware [devices](https://github.com/openvinotoolkit/openvino?tab=readme-ov-file#supported-hardware-matrix) including x86 and ARM CPUs, and Intel GPUs. It can help to boost deep learning performance in Computer Vision, Automatic Speech Recognition, Natural Language Processing and other common tasks.

Hugging Face embedding model can be supported by OpenVINO through ``OpenVINOEmbeddings`` class. If you have an Intel GPU, you can specify `model_kwargs={"device": "GPU"}` to run inference on it.
"""
logger.info("# OpenVINO")

# %pip install --upgrade-strategy eager "optimum[openvino,nncf]" --quiet


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "CPU"}
encode_kwargs = {"mean_pooling": True, "normalize_embeddings": True}

ov_embeddings = OpenVINOEmbeddings(
    model_name_or_path=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

text = "This is a test document."

query_result = ov_embeddings.embed_query(text)

query_result[:3]

doc_result = ov_embeddings.embed_documents([text])

"""
## Export IR model
It is possible to export your embedding model to the OpenVINO IR format with ``OVModelForFeatureExtraction``, and load the model from local folder.
"""
logger.info("## Export IR model")


ov_model_dir = "all-mpnet-base-v2-ov"
if not Path(ov_model_dir).exists():
    ov_embeddings.save_model(ov_model_dir)

ov_embeddings = OpenVINOEmbeddings(
    model_name_or_path=ov_model_dir,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

"""
## BGE with OpenVINO
We can also access BGE embedding models via the ``OpenVINOBgeEmbeddings`` class with OpenVINO.
"""
logger.info("## BGE with OpenVINO")


model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "CPU"}
encode_kwargs = {"normalize_embeddings": True}
ov_embeddings = OpenVINOBgeEmbeddings(
    model_name_or_path=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

embedding = ov_embeddings.embed_query("hi this is harrison")
len(embedding)

"""
For more information refer to:

* [OpenVINO LLM guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html).

* [OpenVINO Documentation](https://docs.openvino.ai/2024/home.html).

* [OpenVINO Get Started Guide](https://www.intel.com/content/www/us/en/content-details/819067/openvino-get-started-guide.html).

* [RAG Notebook with LangChain](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-rag-langchain).
"""
logger.info("For more information refer to:")

logger.info("\n\n[DONE]", bright=True)
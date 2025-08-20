from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.postprocessor.openvino_rerank import OpenVINORerank
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/openvino_rerank.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# OpenVINO Rerank

[OpenVINO™](https://github.com/openvinotoolkit/openvino) is an open-source toolkit for optimizing and deploying AI inference. The OpenVINO™ Runtime supports various hardware [devices](https://github.com/openvinotoolkit/openvino?tab=readme-ov-file#supported-hardware-matrix) including x86 and ARM CPUs, and Intel GPUs. It can help to boost deep learning performance in Computer Vision, Automatic Speech Recognition, Natural Language Processing and other common tasks.

Hugging Face rerank model can be supported by OpenVINO through ``OpenVINORerank`` class.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# OpenVINO Rerank")

# %pip install llama-index-postprocessor-openvino-rerank
# %pip install llama-index-embeddings-openvino

# !pip install llama-index

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Download Embedding, Rerank models and LLM
"""
logger.info("## Download Embedding, Rerank models and LLM")


OpenVINOEmbedding.create_and_save_openvino_model(
    "BAAI/bge-small-en-v1.5", "./embedding_ov"
)


OpenVINORerank.create_and_save_openvino_model(
    "BAAI/bge-reranker-large", "./rerank_ov"
)

# !optimum-cli export openvino --model HuggingFaceH4/zephyr-7b-beta --weight-format int4 llm_ov

"""
## Retrieve top 10 most relevant nodes, then filter with OpenVINO Rerank
"""
logger.info("## Retrieve top 10 most relevant nodes, then filter with OpenVINO Rerank")



Settings.embed_model = OpenVINOEmbedding(model_id_or_path="./embedding_ov")
Settings.llm = OpenVINOLLM(model_id_or_path="./llm_ov")

ov_rerank = OpenVINORerank(
    model_id_or_path="./rerank_ov", device="cpu", top_n=2
)

index = VectorStoreIndex.from_documents(documents=documents)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[ov_rerank],
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

logger.debug(response)

logger.debug(response.get_formatted_sources(length=200))

"""
### Directly retrieve top 2 most similar nodes
"""
logger.info("### Directly retrieve top 2 most similar nodes")

query_engine = index.as_query_engine(
    similarity_top_k=2,
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)

"""
Retrieved context is irrelevant and response is hallucinated.
"""
logger.info("Retrieved context is irrelevant and response is hallucinated.")

logger.debug(response)

logger.debug(response.get_formatted_sources(length=200))

"""
For more information refer to:

* [OpenVINO LLM guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html).

* [OpenVINO Documentation](https://docs.openvino.ai/2024/home.html).

* [OpenVINO Get Started Guide](https://www.intel.com/content/www/us/en/content-details/819067/openvino-get-started-guide.html).

* [RAG example with LlamaIndex](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-rag-llamaindex).
"""
logger.info("For more information refer to:")

logger.info("\n\n[DONE]", bright=True)
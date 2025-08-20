from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openvino_genai import OpenVINOGenAILLM
import huggingface_hub as hf_hub
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/openvino.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# OpenVINO GenAI LLMs

[OpenVINO™](https://github.com/openvinotoolkit/openvino) is an open-source toolkit for optimizing and deploying AI inference. OpenVINO™ Runtime can enable running the same model optimized across various hardware [devices](https://github.com/openvinotoolkit/openvino?tab=readme-ov-file#supported-hardware-matrix). Accelerate your deep learning performance across use cases like: language + LLMs, computer vision, automatic speech recognition, and more.

`OpenVINOGenAILLM` is a wrapper of [OpenVINO-GenAI API](https://github.com/openvinotoolkit/openvino.genai). OpenVINO models can be run locally through this entitiy wrapped by LlamaIndex :

In the below line, we install the packages necessary for this demo:
"""
logger.info("# OpenVINO GenAI LLMs")

# %pip install llama-index-llms-openvino-genai

# %pip install optimum[openvino]

"""
Now that we're set up, let's play around:

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("Now that we're set up, let's play around:")

# !pip install llama-index


"""
### Model Exporting

It is possible to [export your model](https://github.com/huggingface/optimum-intel?tab=readme-ov-file#export) to the OpenVINO IR format with the CLI, and load the model from local folder.
"""
logger.info("### Model Exporting")

# !optimum-cli export openvino --model microsoft/Phi-3-mini-4k-instruct --task text-generation-with-past --weight-format int4 model_path

"""
You can download a optimized IR model from OpenVINO model hub of Hugging Face.
"""
logger.info("You can download a optimized IR model from OpenVINO model hub of Hugging Face.")


model_id = "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"
model_path = "Phi-3-mini-4k-instruct-int4-ov"

hf_hub.snapshot_download(model_id, local_dir=model_path)

"""
### Model Loading

Models can be loaded by specifying the model parameters using the `OpenVINOGenAILLM` method.

If you have an Intel GPU, you can specify `device="gpu"` to run inference on it.
"""
logger.info("### Model Loading")

ov_llm = OpenVINOGenAILLM(
    model_path=model_path,
    device="CPU",
)

"""
You can pass the generation config parameters through `ov_llm.config`. The supported parameters are listed at the [openvino_genai.GenerationConfig](https://docs.openvino.ai/2024/api/genai_api/_autosummary/openvino_genai.GenerationConfig.html).
"""
logger.info("You can pass the generation config parameters through `ov_llm.config`. The supported parameters are listed at the [openvino_genai.GenerationConfig](https://docs.openvino.ai/2024/api/genai_api/_autosummary/openvino_genai.GenerationConfig.html).")

ov_llm.config.max_new_tokens = 100

response = ov_llm.complete("What is the meaning of life?")
logger.debug(str(response))

"""
### Streaming

Using `stream_complete` endpoint
"""
logger.info("### Streaming")

response = ov_llm.stream_complete("Who is Paul Graham?")
for r in response:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = ov_llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

"""
For more information refer to:

* [OpenVINO LLM guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html).

* [OpenVINO Documentation](https://docs.openvino.ai/2024/home.html).

* [OpenVINO Get Started Guide](https://www.intel.com/content/www/us/en/content-details/819067/openvino-get-started-guide.html).

* [RAG example with LlamaIndex](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-rag-llamaindex).
"""
logger.info("For more information refer to:")

logger.info("\n\n[DONE]", bright=True)
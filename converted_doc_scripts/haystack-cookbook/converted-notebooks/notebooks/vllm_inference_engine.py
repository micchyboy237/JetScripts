from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Use the ⚡ vLLM inference engine with Haystack

<img src="https://haystack.deepset.ai/images/haystack-ogimage.png" width="430" style="display:inline;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://docs.vllm.ai/en/latest/_images/vllm-logo-text-light.png" width="350" style="display:inline;">

*Notebook by [Stefano Fiorucci](https://github.com/anakin87)*

This notebook shows how to use the [vLLM inference engine](https://docs.vllm.ai/en/latest/) with Haystack.

## Install vLLM + Haystack

- we install vLLM using pip ([docs](https://docs.vllm.ai/en/latest/getting_started/installation.html))
- for production use cases, there are many other options, including Docker ([docs](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html))
"""
logger.info("# Use the ⚡ vLLM inference engine with Haystack")

# ! nvcc --version

# ! pip install vllm haystack-ai

"""
## Run a vLLM OllamaFunctionCallingAdapter-compatible server in Colab

vLLM can be deployed as a server that implements the OllamaFunctionCallingAdapter API protocol. This allows vLLM to be used as a drop-in replacement for applications using OllamaFunctionCallingAdapter API. Read more [in the docs](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server).

In Colab, we start the OllamaFunctionCallingAdapter-compatible server using Python.
For environments that support Docker, we can run the server using Docker ([docs](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)).

*Significant parameters:*
- **model**: [TheBloke/notus-7B-v1-AWQ](https://huggingface.co/TheBloke/notus-7B-v1-AWQ) is the AWQ quantized version of a good LLM by Argilla. Several model architectures are supported; models are automatically downloaded from Hugging Face as needed. For a comprehensive list of the supported models, see the [docs](https://docs.vllm.ai/en/latest/models/supported_models.html).

- **quantization**: awq. AWQ is a quantization method that allows LLMs to run (fast) when GPU resources are limited. [Simple blogpost on quantization techniques](https://www.maartengrootendorst.com/blog/quantization/#awq-activation-aware-weight-quantization)
- **max_model_len**: we specify a [maximum context length](https://docs.vllm.ai/en/latest/models/engine_args.html), which consists of the maximum number of tokens (prompt + response). Otherwise, the model does not fit in Colab and we get an OOM error.
"""
logger.info("## Run a vLLM OllamaFunctionCallingAdapter-compatible server in Colab")

# ! nohup python -m vllm.entrypoints.openai.api_server \
                  --model TheBloke/notus-7B-v1-AWQ \
                  --quantization awq \
                  --max-model-len 2048 \
                  > vllm.log &

# !while ! grep -q "Application startup complete" vllm.log; do tail -n 1 vllm.log; sleep 5; done

"""
## Chat with the model using OllamaFunctionCallingAdapterChatGenerator

Once we have launched the vLLM-compatible OllamaFunctionCallingAdapter server,
we can simply initialize an `OllamaFunctionCallingAdapterChatGenerator` pointing to the vLLM server URL and start chatting!
"""
logger.info("## Chat with the model using OllamaFunctionCallingAdapterChatGenerator")


generator = OllamaFunctionCallingAdapterChatGenerator(
    api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),  # for compatibility with the OllamaFunctionCallingAdapter API, a placeholder api_key is needed
    model="TheBloke/notus-7B-v1-AWQ",
    api_base_url="http://localhost:8000/v1",
    generation_kwargs = {"max_tokens": 512}
)

messages = []

while True:
  msg = input("Enter your message or Q to exit\n🧑 ")
  if msg=="Q":
    break
  messages.append(ChatMessage.from_user(msg))
  response = generator.run(messages=messages)
  assistant_resp = response['replies'][0]
  logger.debug("🤖 "+assistant_resp.text)
  messages.append(assistant_resp)

logger.info("\n\n[DONE]", bright=True)
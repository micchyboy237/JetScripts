from jet.logger import CustomLogger
from llama_index.llms.nvidia_tensorrt import LocalTensorRTLLM
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/nvidia_tensorrt.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Nvidia TensorRT-LLM

TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs.

[TensorRT-LLM Github](https://github.com/NVIDIA/TensorRT-LLM)

## TensorRT-LLM Environment Setup
Since TensorRT-LLM is a SDK for interacting with local models in process there are a few environment steps that must be followed to ensure that the TensorRT-LLM setup can be used. Please note, that Nvidia Cuda 12.2 or higher is currently required to run TensorRT-LLM.

In this tutorial we will show how to use the connector with GPT2 model.
For the best experience, we recommend following
[Installation](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.8.0?tab=readme-ov-file#installation) process on the
official [TensorRT-LLM Github](https://github.com/NVIDIA/TensorRT-LLM).

The following steps are showing how to set up your model with TensorRT-LLM v0.8.0 for x86_64 users.

1. Obtain and start the basic docker image environment.
```
docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.1.0-devel-ubuntu22.04
```

2. Install dependencies, TensorRT-LLM requires Python 3.10
```
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs wget
```
3. Install the latest stable version (corresponding to the release branch) of TensorRT-LLM. We are using version 0.8.0, but for the most up to date release,
please refer to [official release page](https://github.com/NVIDIA/TensorRT-LLM/releases).
```
pip3 install tensorrt_llm==0.8.0 -U --extra-index-url https://pypi.nvidia.com
```

4. Check installation
```
python3 -c "import tensorrt_llm"
```
The above command should not produce any errors.

5. For this example we will use GPT2. The GPT2 model files need to be created via scripts following the instructions [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt#usage)
    * First, inside the container, we've started during stage 1, clone TensorRT-LLM repository:
    ```
    git clone --branch v0.8.0 https://github.com/NVIDIA/TensorRT-LLM.git
    ```
    * Install requirements for GPT2 model with:
    ```
    cd TensorRT-LLM/examples/gpt/ && pip install -r requirements.txt
    ```
    * Download hf gpt2 model
    ```
    rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2
    cd gpt2
    rm pytorch_model.bin model.safetensors
    wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin
    cd ..
    ```
    * Convert weights from HF Transformers to TensorRT-LLM format
    ```
    python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16
    ```
    * Build TensorRT engine
    ```
    python3 build.py --model_dir=./c-model/gpt2/1-gpu --use_gpt_attention_plugin --remove_input_padding
    ```
  
6. Install `llama-index-llms-nvidia-tensorrt` package
  ```
  pip install llama-index-llms-nvidia-tensorrt
  ```

## Basic Usage

#### Call `complete` with a prompt

```python

llm = LocalTensorRTLLM(
    model_path="./engine_outputs",
    engine_name="gpt_float16_tp1_rank0.engine",
    tokenizer_dir="gpt2",
    max_new_tokens=40,
)

resp = llm.complete("Who is Harry Potter?")
logger.debug(str(resp))
```

The expected response should look like:
```
Harry Potter is a fictional character created by J.K. Rowling in her first novel, Harry Potter and the Philosopher's Stone. The character is a wizard who lives in the fictional town#
```
"""
logger.info("# Nvidia TensorRT-LLM")

logger.info("\n\n[DONE]", bright=True)
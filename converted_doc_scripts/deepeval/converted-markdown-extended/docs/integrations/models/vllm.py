from jet.logger import logger
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
---
# id: vllm
title: vLLM
sidebar_label: vLLM
---

`vLLM` is a high-performance inference engine for LLMs that supports Ollama-compatible APIs. `deepeval` can connect to a running `vLLM` server for running local evaluations.

### Command Line

1. Launch your `vLLM` server and ensure it’s exposing the Ollama-compatible API. The typical base URL for a local vLLM server is: `http://localhost:8000/v1/`.
2. Then run the following command to configure `deepeval`:
"""
logger.info("# id: vllm")

deepeval set-local-model --model-name=<model_name> \
    --base-url="http://localhost:8000/v1/" \
    --api-key=<api-key>

"""
:::tip
You can use any value for `--api-key` if authentication is not enforced.
:::

:::tip Persisting settings
You can persist CLI settings with the optional `--save` flag.
See [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save).
:::

### Reverting to Ollama

To disable the local model and return to Ollama:
"""
logger.info("### Reverting to Ollama")

deepeval unset-local-model

"""
:::info
For advanced setup or deployment options (e.g. multi-GPU, HuggingFace models), see the [vLLM documentation](https://vllm.ai/).
:::
"""
logger.info("For advanced setup or deployment options (e.g. multi-GPU, HuggingFace models), see the [vLLM documentation](https://vllm.ai/).")

logger.info("\n\n[DONE]", bright=True)
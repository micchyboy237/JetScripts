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
# id: lmstudio
title: LM Studio
sidebar_label: LM Studio
---

`deepeval` supports running evaluations using local LLMs that expose Ollama-compatible APIs. One such provider is **LM Studio**, a user-friendly desktop app for running models locally.

### Command Line

To start using LM Studio with `deepeval`, follow these steps:

1. Make sure LM Studio is running. The typical base URL for LM Studio is: `http://localhost:1234/v1/`.
2. Run the following command in your terminal to connect `deepeval` to LM Studio:
"""
logger.info("# id: lmstudio")

deepeval set-local-model --model-name=<model_name> \
    --base-url="http://localhost:1234/v1/" \
    --api-key=<api-key>

"""
:::tip
Use any placeholder string for `--api-key` if your local endpoint doesn't require authentication.
:::

:::tip Persisting settings
You can persist CLI settings with the optional `--save` flag.
See [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save).
:::

### Reverting to Ollama

To switch back to using Ollama’s hosted models, run:
"""
logger.info("### Reverting to Ollama")

deepeval unset-local-model

"""
:::info
For more help on enabling LM Studio’s server or configuring models, check out the [LM Studio docs](https://lmstudio.ai/).
:::
"""
logger.info("For more help on enabling LM Studio’s server or configuring models, check out the [LM Studio docs](https://lmstudio.ai/).")

logger.info("\n\n[DONE]", bright=True)
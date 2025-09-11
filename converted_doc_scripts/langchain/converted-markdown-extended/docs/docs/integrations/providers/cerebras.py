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
# Cerebras

At Cerebras, we've developed the world's largest and fastest AI processor, the Wafer-Scale Engine-3 (WSE-3). The Cerebras CS-3 system, powered by the WSE-3, represents a new class of AI supercomputer that sets the standard for generative AI training and inference with unparalleled performance and scalability.

With Cerebras as your inference provider, you can:
- Achieve unprecedented speed for AI inference workloads
- Build commercially with high throughput
- Effortlessly scale your AI workloads with our seamless clustering technology

Our CS-3 systems can be quickly and easily clustered to create the largest AI supercomputers in the world, making it simple to place and run the largest models. Leading corporations, research institutions, and governments are already using Cerebras solutions to develop proprietary models and train popular open-source models.

Want to experience the power of Cerebras? Check out our [website](https://cerebras.ai) for more resources and explore options for accessing our technology through the Cerebras Cloud or on-premise deployments!

For more information about Cerebras Cloud, visit [cloud.cerebras.ai](https://cloud.cerebras.ai/). Our API reference is available at [inference-docs.cerebras.ai](https://inference-docs.cerebras.ai/).

## Installation and Setup
Install the integration package:
"""
logger.info("# Cerebras")

pip install langchain-cerebras

"""
## API Key
Get an API Key from [cloud.cerebras.ai](https://cloud.cerebras.ai/) and add it to your environment variables:

export CEREBRAS_API_KEY="your-api-key-here"

## Chat Model
See a [usage example](/docs/integrations/chat/cerebras).
"""
logger.info("## API Key")

logger.info("\n\n[DONE]", bright=True)
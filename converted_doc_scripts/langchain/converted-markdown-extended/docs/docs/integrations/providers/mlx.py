from jet.logger import logger
from langchain_community.chat_models.mlx import ChatMLX
from langchain_community.llms.mlx_pipeline import MLXPipeline
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
# MLX

>[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is a `NumPy`-like array framework
> designed for efficient and flexible machine learning on `Apple` silicon,
> brought to you by `Apple machine learning research`.


## Installation and Setup

Install several Python packages:
"""
logger.info("# MLX")

pip install mlx-lm transformers huggingface_hub

"""
## Chat models


See a [usage example](/docs/integrations/chat/mlx).
"""
logger.info("## Chat models")


"""
## LLMs

### MLX Local Pipelines

See a [usage example](/docs/integrations/llms/mlx_pipelines).
"""
logger.info("## LLMs")


logger.info("\n\n[DONE]", bright=True)
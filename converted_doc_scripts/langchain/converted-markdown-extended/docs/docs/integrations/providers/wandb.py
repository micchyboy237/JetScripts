from jet.logger import logger
from langchain_community.callbacks import WandbCallbackHandler
from langchain_community.callbacks import wandb_tracing_enabled
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
# Weights & Biases

>[Weights & Biases](https://wandb.ai/) is provider of the AI developer platform to train and
> fine-tune AI models and develop AI applications.

`Weights & Biase` products can be used to log metrics and artifacts during training,
and to trace the execution of your code.

There are several main ways to use `Weights & Biases` products within LangChain:
- with `wandb_tracing_enabled`
- with `Weave` lightweight toolkit
- with `WandbCallbackHandler` (deprecated)


## wandb_tracing_enabled

See a [usage example](/docs/integrations/providers/wandb_tracing).

See in the [W&B documentation](https://docs.wandb.ai/guides/integrations/langchain).
"""
logger.info("# Weights & Biases")


"""
## Weave

See in the [W&B documentation](https://weave-docs.wandb.ai/guides/integrations/langchain).


## WandbCallbackHandler

**Note:** the `WandbCallbackHandler` is being deprecated in favour of the `wandb_tracing_enabled`.

See a [usage example](/docs/integrations/providers/wandb_tracking).

See in the [W&B documentation](https://docs.wandb.ai/guides/integrations/langchain).
"""
logger.info("## Weave")


logger.info("\n\n[DONE]", bright=True)
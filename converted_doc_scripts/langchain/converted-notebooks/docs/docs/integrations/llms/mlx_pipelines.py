from jet.logger import logger
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_core.prompts import PromptTemplate
from mlx_lm import load
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
# MLX Local Pipelines

MLX models can be run locally through the `MLXPipeline` class.

The [MLX Community](https://huggingface.co/mlx-community) hosts over 150 models, all open source and publicly available on Hugging Face Model Hub a online platform where people can easily collaborate and build ML together.

These can be called from LangChain either through this local pipeline wrapper or by calling their hosted inference endpoints through the MlXPipeline class. For more information on mlx, see the [examples repo](https://github.com/ml-explore/mlx-examples/tree/main/llms) notebook.

To use, you should have the ``mlx-lm`` python [package installed](https://pypi.org/project/mlx-lm/), as well as [transformers](https://pypi.org/project/transformers/). You can also install `huggingface_hub`.
"""
logger.info("# MLX Local Pipelines")

# %pip install --upgrade --quiet  mlx-lm transformers huggingface_hub

"""
### Model Loading

Models can be loaded by specifying the model parameters using the `from_model_id` method.
"""
logger.info("### Model Loading")


pipe = MLXPipeline.from_model_id(
    "mlx-community/quantized-gemma-2b-it",
    pipeline_kwargs={"max_tokens": 10, "temp": 0.1},
)

"""
They can also be loaded by passing in an existing `transformers` pipeline directly
"""
logger.info("They can also be loaded by passing in an existing `transformers` pipeline directly")


model, tokenizer = load("mlx-community/quantized-gemma-2b-it")
pipe = MLXPipeline(model=model, tokenizer=tokenizer)

"""
### Create Chain

With the model loaded into memory, you can compose it with a prompt to
form a chain.
"""
logger.info("### Create Chain")


template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | pipe

question = "What is electroencephalography?"

logger.debug(chain.invoke({"question": question}))

logger.info("\n\n[DONE]", bright=True)
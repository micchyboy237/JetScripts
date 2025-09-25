from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
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
id: synthesizer-generate-from-scratch
title: Generate Goldens From Scratch
sidebar_label: From Scratch
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/synthesizer-generate-from-scratch"
  />
</head>

You can also generate **synthetic Goldens from scratch**, without needing any documents or contexts.

<div
  style={{
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  }}
>
  <img
    src="https://deepeval-docs.s3.amazonaws.com/synthesize-from-scratch.svg"
    style={{
      marginTop: "20px",
      marginBottom: "50px",
      height: "auto",
      maxHeight: "800px",
    }}
  />
</div>

:::info
This approach is particularly useful if your LLM application **doesn't rely on RAG** or if you want to **test your LLM on queries beyond the existing knowledge base**.
:::

## Generate Your Goldens

Since there is no grounded context involved, you'll need to provide a `StylingConfig` when instantiating a `Synthesizer` for `deepeval`'s `Synthesizer` to know what types of goldens it should generate:
"""
logger.info("## Generate Your Goldens")


styling_config = StylingConfig(
  input_format="Questions in English that asks for data in database.",
  expected_output_format="SQL query based on the given input",
  task="Answering text-to-SQL-related queries by querying a database and returning the results to users",
  scenario="Non-technical users trying to query a database using plain English.",
)

synthesizer = Synthesizer(styling_config=styling_config)

"""
Finally, to generate synthetic goldens without provided context, simply supply the number of goldens you want generated:
"""
logger.info("Finally, to generate synthetic goldens without provided context, simply supply the number of goldens you want generated:")


...
synthesizer.generate_goldens_from_scratch(num_goldens=25)
logger.debug(synthesizer.synthetic_goldens)

"""
There is **ONE** mandatory parameter when using the `generate_goldens_from_scratch` method:

- `num_goldens`: the number of goldens to generate.
"""
logger.info("There is **ONE** mandatory parameter when using the `generate_goldens_from_scratch` method:")

logger.info("\n\n[DONE]", bright=True)
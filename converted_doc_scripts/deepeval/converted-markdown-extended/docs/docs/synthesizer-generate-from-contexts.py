from deepeval.synthesizer import Synthesizer
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
id: synthesizer-generate-from-contexts
title: Generate Goldens From Contexts
sidebar_label: From Contexts
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/synthesizer-generate-from-contexts"
  />
</head>

If you already have prepared contexts, you can skip document processing. Simply provide these contexts to the Synthesizer, and it will generate the Goldens directly without processing documents.

<div
  style={{
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  }}
>
  <img
    src="https://deepeval-docs.s3.amazonaws.com/synthesize-from-contexts.svg"
    alt="LangChain"
    style={{
      marginTop: "20px",
      marginBottom: "50px",
      height: "auto",
      maxHeight: "800px",
    }}
  />
</div>

:::tip
This is especially helpful if you **already have an embedded knowledge base**. For example, if you have documents parsed and stored in a vector database, you may handle retrieving text chunks yourself.
:::

## Generate Your Goldens

To generate synthetic `Golden`s from documents, simply provide a list of contexts:
"""
logger.info("## Generate Your Goldens")


synthesizer = Synthesizer()
synthesizer.generate_goldens_from_contexts(
    contexts=[
        ["The Earth revolves around the Sun.", "Planets are celestial bodies."],
        ["Water freezes at 0 degrees Celsius.", "The chemical formula for water is H2O."],
    ]
)

"""
There are **ONE** mandatory and **THREE** optional parameters when using the `generate_goldens_from_contexts` method:

- `contexts`: a list of context, where each context is itself a list of strings, ideally sharing a common theme or subject area.
- [Optional] `include_expected_output`: a boolean which when set to `True`, will additionally generate an `expected_output` for each synthetic `Golden`. Defaulted to `True`.
- [Optional] `max_goldens_per_context`: the maximum number of goldens to be generated per context. Defaulted to 2.
- [Optional] `source_files`: a list of strings specifying the source of the contexts. Length of `source_files` **MUST** be the same as the length of `contexts`.

:::tip DID YOU KNOW?
The `generate_goldens_from_docs()` method calls the `generate_goldens_from_contexts()` method under the hood, and the only difference between the two is the `generate_goldens_from_contexts()` method does not contain a [context construction step](synthesizer-generate-from-docs#how-does-context-construction-work), but instead uses the provided contexts directly for generation.
:::
"""
logger.info("There are **ONE** mandatory and **THREE** optional parameters when using the `generate_goldens_from_contexts` method:")

logger.info("\n\n[DONE]", bright=True)
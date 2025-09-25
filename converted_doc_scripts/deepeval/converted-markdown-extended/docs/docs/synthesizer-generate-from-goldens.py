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
id: synthesizer-generate-from-goldens
title: Generate Goldens From Goldens
sidebar_label: From Goldens
---

`deepeval` enables you to **generate synthetic Goldens from an existing set of Goldens**, without requiring any documents or context. This is ideal for quickly expanding or adding more complexity to your evaluation dataset.

<div
  style={{
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  }}
>
  <img
    src="https://deepeval-docs.s3.us-east-1.amazonaws.com/goldens_from_goldens.svg"
    style={{
      marginTop: "20px",
      marginBottom: "50px",
      height: "auto",
      maxHeight: "800px",
    }}
  />
</div>

:::tip
By default, `generate_goldens_from_goldens` extracts `StylingConfig` from your existing Golden, but it is recommended to [provide a `StylingConfig` explicitly](/docs/synthesizer-introduction#styling-options) for better accuracy and consistency.
:::

## Generate Your Goldens

To get started, simply define a `Synthesizer` object and pass in your list of existing Goldens to the `generate_goldens_from_goldens` method.
"""
logger.info("## Generate Your Goldens")


synthesizer = Synthesizer()
synthesizer.generate_goldens_from_goldens(
  goldens=goldens,
  max_goldens_per_golden=2,
  include_expected_output=True,
)

"""
There is **ONE** mandatory and **TWO** optional parameter when using the `generate_goldens_from_goldens` method:

- `goldens`: a list of existing Goldens from which the new Goldens will be generated.
- [Optional] `max_goldens_per_golden`: the maximum number of goldens to be generated per golden. Defaulted to 2.
- [Optional] `include_expected_output`: a boolean which when set to `True`, will additionally generate an `expected_output` for each synthetic `Golden`. Defaulted to `True`.

:::info
If your existing Goldens include `context`, the synthesizer will utilize these contexts to generate synthetic Goldens, ensuring they are grounded in truth. If no context is present, the synthesizer will employ the `generate_from_scratch` method to create additional inputs based on provided inputs.
:::
"""
logger.info("There is **ONE** mandatory and **TWO** optional parameter when using the `generate_goldens_from_goldens` method:")

logger.info("\n\n[DONE]", bright=True)
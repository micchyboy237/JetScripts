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
# Example selectors

:::note Prerequisites

- [Chat models](/docs/concepts/chat_models/)
- [Few-shot prompting](/docs/concepts/few_shot_prompting/)
:::

## Overview

One common prompting technique for achieving better performance is to include examples as part of the prompt. This is known as [few-shot prompting](/docs/concepts/few_shot_prompting).

This gives the [language model](/docs/concepts/chat_models/) concrete examples of how it should behave.
Sometimes these examples are hardcoded into the prompt, but for more advanced situations it may be nice to dynamically select them.

**Example Selectors** are classes responsible for selecting and then formatting examples into prompts.

## Related resources

* [Example selector how-to guides](/docs/how_to/#example-selectors)
"""
logger.info("# Example selectors")

logger.info("\n\n[DONE]", bright=True)
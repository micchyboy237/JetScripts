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
id: miscellaneous
title: Miscellaneous
sidebar_label: Miscellaneous
---

<head>
  <link rel="canonical" href="https://deepeval.com/docs/miscellaneous" />
</head>

Opt-in to update warnings as follows:
"""
logger.info("id: miscellaneous")

export DEEPEVAL_UPDATE_WARNING_OPT_IN=1

"""
It is highly recommended that you opt-in to update warnings.
"""
logger.info("It is highly recommended that you opt-in to update warnings.")

logger.info("\n\n[DONE]", bright=True)
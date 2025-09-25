from deepteam.vulnerabilities import Misinformation
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
id: red-teaming-vulnerabilities-misinformation
title: Misinformation
sidebar_label: Misinformation
---

<head>
  <link
    rel="canonical"
    href="https://trydeepteam.com/docs/red-teaming-vulnerabilities-misinformation"
  />
</head>

The misinformation vulnerability is designed to test whether an LLM can accurately provide information, verify claims, and avoid generating or amplifying false or misleading content.
"""
logger.info("id: red-teaming-vulnerabilities-misinformation")

pip install deepteam

"""

"""


misinformation = Misinformation(types=["factual error"])

"""
Learn more how to red teaming LLM systems using the misinformation vulnerability on [DeepTeam's docs.](https://trydeepteam.com/docs/red-teaming-vulnerabilities-misinformation)

:::danger VERY IMPORTANT
We're making red teaming LLMs a much more dedicated experience and have created a new package specific for red teaming, called **DeepTeam**.

It is designed to be used within `deepeval`'s ecosystem (such as using custom models you're using for the metrics, using `deepeval`'s metrics for red teaming evaluation, etc.).

To begin, install `deepteam`:
"""
logger.info("Learn more how to red teaming LLM systems using the misinformation vulnerability on [DeepTeam's docs.](https://trydeepteam.com/docs/red-teaming-vulnerabilities-misinformation)")

pip install deepteam

"""
You can read [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-vulnerabilities)
:::
"""
logger.info("You can read [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-vulnerabilities)")

logger.info("\n\n[DONE]", bright=True)
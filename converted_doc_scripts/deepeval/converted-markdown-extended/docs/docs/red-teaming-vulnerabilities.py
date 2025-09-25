from deepteam import red_team
from deepteam.vulnerabilities import bias, misinformation
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
id: red-teaming-vulnerabilities
title: Vulnerabilities
sidebar_label: Vulnerabilities
---

<head>
  <link
    rel="canonical"
    href="https://trydeepteam.com/docs/red-teaming-vulnerabilities"
  />
</head>

LLM vulnerabilities such as bias, PII leakage (which can introduced in fine-tuning of during tool calling), misinformation, are all important aspects that require red teaming and detection.

:::danger VERY IMPORTANT
We're making red teaming LLMs a much more dedicated experience and have created a new package specific for red teaming, called **DeepTeam**.

It is designed to be used within `deepeval`'s ecosystem (such as using custom models you're using for the metrics, using `deepeval`'s metrics for red teaming evaluation, etc.).

To begin, install `deepteam`:
"""
logger.info("id: red-teaming-vulnerabilities")

pip install deepteam

"""
You can read [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-vulnerabilities)
:::

Here's how you can select different vulnerabilities to red team your LLM system on using **DeepTeam**:
"""
logger.info("You can read [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-vulnerabilities)")


bias = Bias()
misinformation = Misinformation()

risk_assessment = red_team(vulnerabilities=[bias, misinformation], attacks=..., model_callback=...)

"""
You can read how to use vulnerabilities on [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-vulnerabilities)
"""
logger.info("You can read how to use vulnerabilities on [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-vulnerabilities)")

logger.info("\n\n[DONE]", bright=True)
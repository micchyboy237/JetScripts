from deepteam import red_team
from deepteam.attacks import PromptInjection
from deepteam.vulnerabilities import Bias
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
id: red-teaming-introduction
title: Introduction
sidebar_label: Introduction
---

<head>
  <link
    rel="canonical"
    href="https://trydeepteam.com/docs/red-teaming-introduction"
  />
</head>

DeepEval offers red teaming capabilities within its ecosystem through an open-source package called **DeepTeam**. DeepTeam is DeepEval for red teaming.

:::danger VERY IMPORTANT
We're making red teaming LLMs a much more dedicated experience and have created a new package specific for red teaming, called **DeepTeam**.

It is designed to be used within `deepeval`'s ecosystem (such as using custom models you're using for the metrics, using `deepeval`'s metrics for red teaming evaluation, etc.).

To begin, install `deepteam`:
"""
logger.info("id: red-teaming-introduction")

pip install deepteam

"""
You can read [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-introduction)
:::

Here's how to red team LLMs using DeepTeam:
"""
logger.info("You can read [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-introduction)")

pip install deepteam

"""
Then, paste in the following code:
"""
logger.info("Then, paste in the following code:")


def model_callback(input: str) -> str:
    return f"I'm sorry but I can't answer this: {input}"

bias = Bias(types=["race"])
prompt_injection = PromptInjection()

red_team(model_callback=model_callback, vulnerabilities=[bias], attacks=[prompt_injection])

"""
To learn more on how to red team LLMs, [click here](https://trydeepteam.com/docs/red-teaming-introduction) for the DeepTeam documentation.

:::info
Red teaming, unlike the standard LLM evaluation handled by `deepeval`, is designed to simulate how a malicious user or bad actor might attempt to compromise your systems through your LLM application.
:::
"""
logger.info("To learn more on how to red team LLMs, [click here](https://trydeepteam.com/docs/red-teaming-introduction) for the DeepTeam documentation.")

logger.info("\n\n[DONE]", bright=True)
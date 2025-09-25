from deepteam import red_team
from deepteam.attacks.multi_turn import LinearJailbreaking
from deepteam.attacks.single_turn import PromptInjection
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
id: red-teaming-attack-enhancements
title: Adversarial Attacks
sidebar_label: Adversarial Attacks
---

<head>
  <link
    rel="canonical"
    href="https://trydeepteam.com/docs/red-teaming-adversarial-attacks"
  />
</head>

Adversarial attacks such as **prompt injection, leetspeak, ROT13**, etc. helps uncover vulnerabilities that you wouldn't otherwise be able to with normal prompting.

:::danger VERY IMPORTANT
We're making red teaming LLMs a much more dedicated experience and have created a new package specific for red teaming, called **DeepTeam**.

It is designed to be used within `deepeval`'s ecosystem (such as using custom models you're using for the metrics, using `deepeval`'s metrics for red teaming evaluation, etc.).

To begin, install `deepteam`:
"""
logger.info("id: red-teaming-attack-enhancements")

pip install deepteam

"""
You can read [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-adversarial-attacks)
:::

DeepTeam provides 10+ adversarial attack types (both single and multi-turn attacks) and you can use it easily:
"""
logger.info("You can read [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-adversarial-attacks)")

pip install deepteam

"""

"""


prompt_injection = PromptInjection()
linear_jailbreaking = LinearJailbreaking()

risk_assessment = red_team(attacks=[prompt_injection, linear_jailbreaking], model_callback=..., vulnerabilities=...)

"""
You can read how to use adversarial attacks on [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-adversarial-attacks)
"""
logger.info("You can read how to use adversarial attacks on [DeepTeam's docs here.](https://trydeepteam.com/docs/red-teaming-adversarial-attacks)")

logger.info("\n\n[DONE]", bright=True)
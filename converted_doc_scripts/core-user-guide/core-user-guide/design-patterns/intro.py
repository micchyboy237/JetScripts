from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Intro

Agents can work together in a variety of ways to solve problems.
Research works like [AutoGen](https://aka.ms/autogen-paper),
[MetaGPT](https://arxiv.org/abs/2308.00352)
and [ChatDev](https://arxiv.org/abs/2307.07924) have shown
multi-agent systems out-performing single agent systems at complex tasks
like software development.

A multi-agent design pattern is a structure that emerges from message protocols:
it describes how agents interact with each other to solve problems.
For example, the [tool-equipped agent](../components/tools.ipynb#tool-equipped-agent) in
the previous section employs a design pattern called ReAct,
which involves an agent interacting with tools.

You can implement any multi-agent design pattern using AutoGen agents.
In the next two sections, we will discuss two common design patterns:
group chat for task decomposition, and reflection for robustness.
"""
logger.info("# Intro")

logger.info("\n\n[DONE]", bright=True)
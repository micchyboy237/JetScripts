from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Copilot Agent Architecture Designing - Sep 23, 2024
---

### Speakers: SallyAnn DeLucia, John Gilhuly

### Biography of the speakers:

SallyAnn DeLucia is a Senior AI Product Manager at Arize and a generative AI specialist with a master’s degree in Applied Data Science. She combines a creative outlook with a commitment to developing solutions that are both technically sound and socially responsible. SallyAnn's work emphasizes innovation in AI while maintaining a focus on ethical and practical applications.

John is a developer advocate at Arize AI focused on open-source LLM observability and evaluation tooling. He holds an MBA from Stanford, where he focused on the ethical, social, and business implications of open vs closed AI development, and a B.S. in C.S. from Duke. Prior to joining Arize, John led GTM activities at Slingshot AI, and served as a venture fellow at Omega Venture Partners. In his pre-AI life, John built out and ran technical go-to-market teams at Branch Metrics.

### Abstract:

In this talk, the Arize team will share learnings from their experience building out a fully-featured copilot agent within their product. They'll speak to the various architectural decisions that shaped their agent, and some surprising challenges they ran into along the way. Time permitting, they'll also speak more generally to the state of agents today, and common agent architectures they've seen deployed in market.
"""
logger.info("### Speakers: SallyAnn DeLucia, John Gilhuly")

logger.info("\n\n[DONE]", bright=True)
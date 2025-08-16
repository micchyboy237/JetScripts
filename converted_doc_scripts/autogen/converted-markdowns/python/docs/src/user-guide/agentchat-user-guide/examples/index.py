from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
myst:
  html_meta:
    "description lang=en": |
      Examples built using AgentChat, a high-level api for AutoGen
---

# Examples

A list of examples to help you get started with AgentChat.

:::::{grid} 2 2 2 3

::::{grid-item-card} Travel Planning
:img-top: ../../../images/example-travel.jpeg
:img-alt: travel planning example
:link: ./travel-planning.html
:link-alt: travel planning: Generating a travel plan using multiple agents.

^^^
Generating a travel plan using multiple agents.

::::

::::{grid-item-card} Company Research
:img-top: ../../../images/example-company.jpg
:img-alt: company research example
:link: ./company-research.html
:link-alt: company research: Generating a company research report using multiple agents with tools.

^^^
Generating a company research report using multiple agents with tools.

::::

::::{grid-item-card} Literature Review
:img-top: ../../../images/example-literature.jpg
:img-alt: literature review example
:link: ./literature-review.html
:link-alt: literature review: Generating a literature review using agents with tools.

^^^
Generating a literature review using agents with tools.

::::

:::::
"""
logger.info("# Examples")

:maxdepth: 1
:hidden:

travel-planning
company-research
literature-review

logger.info("\n\n[DONE]", bright=True)
from jet.logger import logger
import LinkCards from "@site/src/components/LinkCards"
import TechStackCards from "@site/src/components/TechStackCards"
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
id: introduction
title: Introduction to Summarizer Evaluation
sidebar_label: Introduction
---


Learn how to build, evaluate, and deploy a reliable **LLM-powered meeting summarization agent** using **Ollama** and **DeepEval**.

<TechStackCards
    techStack={[
        {
            name: "Ollama",
            logo: "https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/light/ollama.png",
        },
        {
            name: "DeepEval",
            logo: "https://pbs.twimg.com/profile_images/1888060560161574912/qbw1-_2g.png",
        }
    ]}
/>

:::note
If you're working with LLMs for summarization, this tutorial is for you. While we'll specifically focus on evaluating a meeting summarizer, the concepts and practices here can be applied to **any LLM application tasked with summary generation**.
:::

## Get Started

DeepEval is an open-source LLM evaluation framework that supports a wide-range of metrics to help evaluate and iterate on your LLM applications.

Click on these links to jump to different stages of this tutorial:

<LinkCards
    tutorials={[
        {
            number: 1,
            icon: "Hammer",
            title: 'Build your Summarizer',
            objectives: [
                "Use Ollama to build a summarizer",
                "Learn modular coding techniques to improve your summarizer",
                "Learn parsing techniques to build production grade LLM applications"
            ],
            to: '/tutorials/summarization-agent/development',
        },
        {
            number: 2,
            icon: "TestTubeDiagonal",
            title: 'Evaluate your summarizer',
            objectives: [
                "Learn how to define your evaluation criteria",
                "Create test cases using your summarizer",
                "Run your first eval",
                "Create datasets for future evaluations"
            ],
            to: '/tutorials/summarization-agent/evaluation',
        },
        {
            number: 3,
            icon: "BookPlus",
            title: 'Changing your model and prompts',
            objectives: [
                "Use evaluation scores to improve your summarizer",
                "Iterate over different models to find the best one for your use case",
                "Change your system prompts and check for regressions"
            ],
            to: '/tutorials/summarization-agent/improvement',
        },
        {
            number: 4,
            title: 'Setup Evals in Production',
            icon:"ShieldCheck",
            objectives: [
                "Trace your entire application workflow",
                "Evaluate your summarizer during prod and choose your metrics",
                "Setup CI/CD workflows to always get reliable summaries"
            ],
            to: '/tutorials/summarization-agent/evals-in-prod',
        },
    ]}
/>

## What You Will Evaluate

In this tutorial you will build and evaluate a **meeting summarization agent** that is used by famous tools like **Otter.ai** and **Circleback** to generate their summaries and action items from meeting transcripts. You will use `deepeval` and evalue the summarization agent's ability to generate:

- A concise summary of the discussion
- A clear list of action items

Below is an example of what a deliverable from a meeting summarization platform might look like:

![Webpage Image](https://deepeval-docs.s3.us-east-1.amazonaws.com/tutorials:summarization-agent:summarizer-overview.png)

In the next section, we'll build this summarization agent from scratch using Ollama API.

:::tip
If you already have an LLM agent to evaluate, you can skip to [Evaluation Section](tutorial-summarization-evaluation) of this tutorial.
:::
"""
logger.info("## Get Started")

logger.info("\n\n[DONE]", bright=True)
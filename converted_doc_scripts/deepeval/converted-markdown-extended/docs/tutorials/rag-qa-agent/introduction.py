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
title: RAG Agent Evaluation Tutorial
sidebar_label: Introduction
---


This tutorial walks you through the entire process of building a reliable **RAG (_Retrieval-Augmented Generation_) QA Agent**, 
from initial development to iterative improvement through `deepeval`'s evaluations.  We'll build this RAG QA Agent using **Ollama**, **LangChain** and **DeepEval**.

<TechStackCards
    techStack={[
        {
            name: "Ollama",
            logo: "https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/light/ollama.png",
        },
        {
            name: "LangChain",
            logo: "https://logo.svgcdn.com/s/langchain-dark-8x.png",
        },
        {
            name: "DeepEval",
            logo: "https://pbs.twimg.com/profile_images/1888060560161574912/qbw1-_2g.png",
        }
    ]}
/>

:::note
This tutorial focuses on building a RAG-based QA agent for an infamous company called **Theranos**. However, the concepts and practices used throughout this tutorial are applicable to any **RAG-based application**. If you are working with RAG applications, this tutorial will be helpful to you.
:::

## Overview

DeepEval is an open-source LLM evaluation framework that supports a wide-range of metrics to help evaluate and iterate on your LLM applications.

You can click on the links below and jump to any stage of this tutorial as you like:

<LinkCards
    tutorials={[
        {
            number: 1,
            icon: "Drill",
            title: 'Develop Your RAG',
            objectives: [
                "Build a RAG with Ollama & LangChain",
                "Use Ollama Embeddings",
                "Use LangChain's vector stores",
                "Create a full RAG QA Agent"
            ],
            to: '/tutorials/rag-qa-agent/development',
        },
        {
            number: 2,
            icon: "TestTubes",
            title: 'Evaluate Your Retriever & Generator',
            objectives: [
                "Define your evaluation criteria",
                "Evaluate your retriever and generator in isolation",
                "Evaluate your RAG as a whole",
                "Create datasets for robust eval pipelines"
            ],
            to: '/tutorials/rag-qa-agent/evaluation',
        },
        {
            number: 3,
            title: 'Improve your RAG using evals',
            icon: "FolderUp",
            objectives: [
                "Define your hyperparamets",
                "Test different configurations with DeepEval",
                "Find the best set of hyperparameters for your RAG"
            ],
            to: '/tutorials/rag-qa-agent/improvement',
        },
        {
            number: 4,
            title: 'Deploy and test your RAG in prod',
            icon: "ShieldAlert",
            objectives: [
                "Trace your RAG components for each QA",
                "Choose the metrics to apply in prod",
                "Test your RAG for every new doc you push in your knowledge base"
            ],
            to: '/tutorials/rag-qa-agent/evals-in-prod',
        },
    ]}
/>

## What You Will Evaluate

**RAG (Retrieval-Augmented Generation)** agents let companies build domain-specific assistants without fine-tuning large models.
In this tutorial, you'll create a **RAG QA agent** that answers questions about **Theranos**, a blood diagnostics company. We will evaluate the agent's ability on:

- Generating relevant and accurate answers
- Providing correct citations to questions

Below is an example of what **Theranos**'s internal RAG QA agent might look like:.


![MadeUpCompany's RAG QA Agent](https://deepeval-docs.s3.us-east-1.amazonaws.com/tutorials:qa-agent:qa-agent-overview.png)

In the following sections of this tutorial, you'll learn how to build a reliable RAG QA Agent that retrieves correct data and generates an 
accurate answer based on the retrieved context.
"""
logger.info("## Overview")

logger.info("\n\n[DONE]", bright=True)
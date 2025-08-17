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
title: '💻 Full stack'
---

Get started with full-stack RAG applications using Embedchain's easy-to-use CLI tool. Set up everything with just a few commands, whether you prefer Docker or not.

## Prerequisites

Choose your setup method:

* [Without docker](#without-docker)
* [With Docker](#with-docker)

### Without Docker

Ensure these are installed:

- Embedchain python package (`pip install embedchain`)
- [Node.js](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) and [Yarn](https://classic.yarnpkg.com/lang/en/docs/install/)

### With Docker

Install Docker from [Docker's official website](https://docs.docker.com/engine/install/).

## Quick Start Guide

### Install the package

Before proceeding, make sure you have the Embedchain package installed.
"""
logger.info("## Prerequisites")

pip install embedchain -U

"""
### Setting Up

# For the purpose of the demo, you have to set `OPENAI_API_KEY` to start with but you can choose any llm by changing the configuration easily.

### Installation Commands

<CodeGroup>
"""
logger.info("### Setting Up")

ec create-app my-app
cd my-app
ec start

"""

"""

ec create-app my-app --docker
cd my-app
ec start --docker

"""
</CodeGroup>

### What Happens Next?

1. Embedchain fetches a full stack template (FastAPI backend, Next.JS frontend).
2. Installs required components.
3. Launches both frontend and backend servers.

### See It In Action

Open http://localhost:3000 to view the chat UI.

![full stack example](/images/fullstack.png)

### Admin Panel

Check out the Embedchain admin panel to see the document chunks for your RAG application.

![full stack chunks](/images/fullstack-chunks.png)

### API Server

If you want to access the API server, you can do so at http://localhost:8000/docs.

![API Server](/images/fullstack-api-server.png)

You can customize the UI and code as per your requirements.
"""
logger.info("### What Happens Next?")

logger.info("\n\n[DONE]", bright=True)
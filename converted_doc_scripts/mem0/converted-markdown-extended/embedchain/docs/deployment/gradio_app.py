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
title: 'Gradio.app'
description: 'Deploy your RAG application to gradio.app platform'
---

Embedchain offers a Streamlit template to facilitate the development of RAG chatbot applications in just three easy steps.

Follow the instructions given below to deploy your first application quickly:

## Step-1: Create RAG app

We provide a command line utility called `ec` in embedchain that inherits the template for `gradio.app` platform and help you deploy the app. Follow the instructions to create a gradio.app app using the template provided:
"""
logger.info("## Step-1: Create RAG app")

pip install embedchain

"""

"""

mkdir my-rag-app
ec create --template=gradio.app

"""
This will generate a directory structure like this:
"""
logger.info("This will generate a directory structure like this:")

├── app.py
├── embedchain.json
└── requirements.txt

"""
Feel free to edit the files as required.
- `app.py`: Contains API app code
- `embedchain.json`: Contains embedchain specific configuration for deployment (you don't need to configure this)
- `requirements.txt`: Contains python dependencies for your application

## Step-2: Test app locally

You can run the app locally by simply doing:
"""
logger.info("## Step-2: Test app locally")

pip install -r requirements.txt
ec dev

"""
## Step-3: Deploy to gradio.app
"""
logger.info("## Step-3: Deploy to gradio.app")

ec deploy

"""
This will run `gradio deploy` which will prompt you questions and deploy your app directly to huggingface spaces.

<img src="/images/gradio_app.png" alt="gradio app" />

## Seeking help?

If you run into issues with deployment, please feel free to reach out to us via any of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Seeking help?")

logger.info("\n\n[DONE]", bright=True)
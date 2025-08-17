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
title: 'Huggingface.co'
description: 'Deploy your RAG application to huggingface.co platform'
---

With Embedchain, you can directly host your apps in just three steps to huggingface spaces where you can view and deploy your app to the world.

We support two types of deployment to huggingface spaces:

<CardGroup cols={2}>
    <Card title="" href="#using-streamlit-io">
        Streamlit.io
    </Card>
    <Card title="" href="#using-gradio-app">
        Gradio.app
    </Card>
</CardGroup>

## Using streamlit.io

### Step 1: Create a new RAG app

Create a new RAG app using the following command:
"""
logger.info("## Using streamlit.io")

mkdir my-rag-app
ec create --template=hf/streamlit.io # inside my-rag-app directory

"""
When you run this for the first time, you'll be asked to login to huggingface.co. Once you login, you'll need to create a **write** token. You can create a write token by going to [huggingface.co settings](https://huggingface.co/settings/token). Once you create a token, you'll be asked to enter the token in the terminal.

This will also create an `embedchain.json` file in your app directory. Add a `name` key into the `embedchain.json` file. This will be the "repo-name" of your app in huggingface spaces.
"""
logger.info("When you run this for the first time, you'll be asked to login to huggingface.co. Once you login, you'll need to create a **write** token. You can create a write token by going to [huggingface.co settings](https://huggingface.co/settings/token). Once you create a token, you'll be asked to enter the token in the terminal.")

{
    "name": "my-rag-app",
    "provider": "hf/streamlit.io"
}

"""
### Step-2: Test app locally

You can run the app locally by simply doing:
"""
logger.info("### Step-2: Test app locally")

pip install -r requirements.txt
ec dev

"""
### Step-3: Deploy to huggingface spaces
"""
logger.info("### Step-3: Deploy to huggingface spaces")

ec deploy

"""
This will deploy your app to huggingface spaces. You can view your app at `https://huggingface.co/spaces/<your-username>/my-rag-app`. This will get prompted in the terminal once the app is deployed.

## Using gradio.app

Similar to streamlit.io, you can deploy your app to gradio.app in just three steps.

### Step 1: Create a new RAG app

Create a new RAG app using the following command:
"""
logger.info("## Using gradio.app")

mkdir my-rag-app
ec create --template=hf/gradio.app # inside my-rag-app directory

"""
When you run this for the first time, you'll be asked to login to huggingface.co. Once you login, you'll need to create a **write** token. You can create a write token by going to [huggingface.co settings](https://huggingface.co/settings/token). Once you create a token, you'll be asked to enter the token in the terminal.

This will also create an `embedchain.json` file in your app directory. Add a `name` key into the `embedchain.json` file. This will be the "repo-name" of your app in huggingface spaces.
"""
logger.info("When you run this for the first time, you'll be asked to login to huggingface.co. Once you login, you'll need to create a **write** token. You can create a write token by going to [huggingface.co settings](https://huggingface.co/settings/token). Once you create a token, you'll be asked to enter the token in the terminal.")

{
    "name": "my-rag-app",
    "provider": "hf/gradio.app"
}

"""
### Step-2: Test app locally

You can run the app locally by simply doing:
"""
logger.info("### Step-2: Test app locally")

pip install -r requirements.txt
ec dev

"""
### Step-3: Deploy to huggingface spaces
"""
logger.info("### Step-3: Deploy to huggingface spaces")

ec deploy

"""
This will deploy your app to huggingface spaces. You can view your app at `https://huggingface.co/spaces/<your-username>/my-rag-app`. This will get prompted in the terminal once the app is deployed.

## Seeking help?

If you run into issues with deployment, please feel free to reach out to us via any of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Seeking help?")

logger.info("\n\n[DONE]", bright=True)
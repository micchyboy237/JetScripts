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
title: 'Streamlit.io'
description: 'Deploy your RAG application to streamlit.io platform'
---

Embedchain offers a Streamlit template to facilitate the development of RAG chatbot applications in just three easy steps.

Follow the instructions given below to deploy your first application quickly:

## Step-1: Create RAG app

We provide a command line utility called `ec` in embedchain that inherits the template for `streamlit.io` platform and help you deploy the app. Follow the instructions to create a streamlit.io app using the template provided:
"""
logger.info("## Step-1: Create RAG app")

pip install embedchain

"""

"""

mkdir my-rag-app
ec create --template=streamlit.io

"""
This will generate a directory structure like this:
"""
logger.info("This will generate a directory structure like this:")

├── .streamlit
│   └── secrets.toml
├── app.py
├── embedchain.json
└── requirements.txt

"""
Feel free to edit the files as required.
- `app.py`: Contains API app code
- `.streamlit/secrets.toml`: Contains secrets for your application
- `embedchain.json`: Contains embedchain specific configuration for deployment (you don't need to configure this)
- `requirements.txt`: Contains python dependencies for your application

# Add your `OPENAI_API_KEY` in `.streamlit/secrets.toml` file to run and deploy the app.

## Step-2: Test app locally

You can run the app locally by simply doing:
"""
logger.info("## Step-2: Test app locally")

pip install -r requirements.txt
ec dev

"""
## Step-3: Deploy to streamlit.io

![Streamlit App deploy button](https://github.com/embedchain/embedchain/assets/73601258/90658e28-29e5-4ceb-9659-37ff8b861a29)

Use the deploy button from the streamlit website to deploy your app.

You can refer this [guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app) if you run into any problems.

## Seeking help?

If you run into issues with deployment, please feel free to reach out to us via any of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Step-3: Deploy to streamlit.io")

logger.info("\n\n[DONE]", bright=True)
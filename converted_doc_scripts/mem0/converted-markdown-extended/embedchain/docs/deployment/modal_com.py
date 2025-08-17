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
title: 'Modal.com'
description: 'Deploy your RAG application to modal.com platform'
---

Embedchain has a nice and simple abstraction on top of the [Modal.com](https://modal.com/) tools to let developers deploy RAG application to modal.com platform seamlessly. 

Follow the instructions given below to deploy your first application quickly:


## Step-1 Create RAG application: 

We provide a command line utility called `ec` in embedchain that inherits the template for `modal.com` platform and help you deploy the app. Follow the instructions to create a modal.com app using the template provided:
"""
logger.info("## Step-1 Create RAG application:")

pip install embedchain[modal]
mkdir my-rag-app
ec create --template=modal.com

"""
This `create` command will open a browser window and ask you to login to your modal.com account and will generate a directory structure like this:
"""
logger.info("This `create` command will open a browser window and ask you to login to your modal.com account and will generate a directory structure like this:")

├── app.py
├── .env
├── .env.example
├── embedchain.json
└── requirements.txt

"""
Feel free to edit the files as required.
- `app.py`: Contains API app code
- `.env`: Contains environment variables for production
- `.env.example`: Contains dummy environment variables (can ignore this file)
- `embedchain.json`: Contains embedchain specific configuration for deployment (you don't need to configure this)
- `requirements.txt`: Contains python dependencies for your FastAPI application

## Step-2: Test app locally

You can run the app locally by simply doing:
"""
logger.info("## Step-2: Test app locally")

pip install -r requirements.txt
ec dev

"""
## Step-3: Deploy to modal.com

You can deploy to modal.com using the following command:
"""
logger.info("## Step-3: Deploy to modal.com")

ec deploy

"""
Once this step finished, it will provide you with the deployment endpoint where you can access the app live. It will look something like this (Swagger docs):

<img src="/images/fly_io.png" />

## Seeking help?

If you run into issues with deployment, please feel free to reach out to us via any of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Seeking help?")

logger.info("\n\n[DONE]", bright=True)
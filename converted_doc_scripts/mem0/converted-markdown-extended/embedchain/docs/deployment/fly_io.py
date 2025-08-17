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
title: 'Fly.io'
description: 'Deploy your RAG application to fly.io platform'
---

Embedchain has a nice and simple abstraction on top of the [Fly.io](https://fly.io/) tools to let developers deploy RAG application to fly.io platform seamlessly. 

Follow the instructions given below to deploy your first application quickly:


## Step-1: Install flyctl command line

<CodeGroup>
"""
logger.info("## Step-1: Install flyctl command line")

brew install flyctl

"""

"""

curl -L https://fly.io/install.sh | sh

"""

"""

pwsh -Command "iwr https://fly.io/install.ps1 -useb | iex"

"""
</CodeGroup>

Once you have installed the fly.io cli tool, signup/login to their platform using the following command:

<CodeGroup>
"""
logger.info("Once you have installed the fly.io cli tool, signup/login to their platform using the following command:")

fly auth signup

"""

"""

fly auth login

"""
</CodeGroup>

In case you run into issues, refer to official [fly.io docs](https://fly.io/docs/hands-on/install-flyctl/).

## Step-2: Create RAG app

We provide a command line utility called `ec` in embedchain that inherits the template for `fly.io` platform and help you deploy the app. Follow the instructions to create a fly.io app using the template provided:
"""
logger.info("## Step-2: Create RAG app")

pip install embedchain

"""

"""

mkdir my-rag-app
ec create --template=fly.io

"""
This will generate a directory structure like this:
"""
logger.info("This will generate a directory structure like this:")

├── Dockerfile
├── app.py
├── fly.toml
├── .env
├── .env.example
├── embedchain.json
└── requirements.txt

"""
Feel free to edit the files as required.
- `Dockerfile`: Defines the steps to setup the application
- `app.py`: Contains API app code
- `fly.toml`: fly.io config file
- `.env`: Contains environment variables for production
- `.env.example`: Contains dummy environment variables (can ignore this file)
- `embedchain.json`: Contains embedchain specific configuration for deployment (you don't need to configure this)
- `requirements.txt`: Contains python dependencies for your application

## Step-3: Test app locally

You can run the app locally by simply doing:
"""
logger.info("## Step-3: Test app locally")

pip install -r requirements.txt
ec dev

"""
## Step-4: Deploy to fly.io

You can deploy to fly.io using the following command:
"""
logger.info("## Step-4: Deploy to fly.io")

ec deploy

"""
Once this step finished, it will provide you with the deployment endpoint where you can access the app live. It will look something like this (Swagger docs):

You can also check the logs, monitor app status etc on their dashboard by running command `fly dashboard`.

<img src="/images/fly_io.png" />

## Seeking help?

If you run into issues with deployment, please feel free to reach out to us via any of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Seeking help?")

logger.info("\n\n[DONE]", bright=True)
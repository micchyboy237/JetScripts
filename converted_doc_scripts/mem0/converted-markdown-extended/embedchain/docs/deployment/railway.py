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
title: 'Railway.app'
description: 'Deploy your RAG application to railway.app'
---

It's easy to host your Embedchain-powered apps and APIs on railway.

Follow the instructions given below to deploy your first application quickly:

## Step-1: Create RAG app
"""
logger.info("## Step-1: Create RAG app")

pip install embedchain

"""
<Tip>
**Create a full stack app using Embedchain CLI**

To use your hosted embedchain RAG app, you can easily set up a FastAPI server that can be used anywhere.
To easily set up a FastAPI server, check out [Get started with Full stack](https://docs.embedchain.ai/get-started/full-stack) page.

Hosting this server on railway is super easy!

</Tip>

## Step-2: Set up your project

### With Docker

You can create a `Dockerfile` in the root of the project, with all the instructions. However, this method is sometimes slower in deployment.

### Without Docker

By default, Railway uses Python 3.7. Embedchain requires the python version to be >3.9 in order to install.

To fix this, create a `.python-version` file in the root directory of your project and specify the correct version
"""
logger.info("## Step-2: Set up your project")

3.10

"""
You also need to create a `requirements.txt` file to specify the requirements.
"""
logger.info("You also need to create a `requirements.txt` file to specify the requirements.")

python-dotenv
embedchain
fastapi==0.108.0
uvicorn==0.25.0
embedchain
beautifulsoup4
sentence-transformers

"""
## Step-3: Deploy to Railway 🚀

1. Go to https://railway.app and create an account.
2. Create a project by clicking on the "Start a new project" button

### With Github

Select `Empty Project` or `Deploy from Github Repo`. 

You should be all set!

### Without Github

You can also use the railway CLI to deploy your apps from the terminal, if you don't want to connect a git repository.

To do this, just run this command in your terminal
"""
logger.info("## Step-3: Deploy to Railway 🚀")

npm i -g @railway/cli
railway login
railway link [projectID]

"""
Finally, run `railway up` to deploy your app.
"""
logger.info("Finally, run `railway up` to deploy your app.")

railway up

"""
## Seeking help?

If you run into issues with deployment, please feel free to reach out to us via any of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Seeking help?")

logger.info("\n\n[DONE]", bright=True)
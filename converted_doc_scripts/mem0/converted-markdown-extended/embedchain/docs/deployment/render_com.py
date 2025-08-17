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
title: 'Render.com'
description: 'Deploy your RAG application to render.com platform'
---

Embedchain has a nice and simple abstraction on top of the [render.com](https://render.com/) tools to let developers deploy RAG application to render.com platform seamlessly. 

Follow the instructions given below to deploy your first application quickly:

## Step-1: Install `render` command line

<CodeGroup>
"""
logger.info("## Step-1: Install `render` command line")

brew tap render-oss/render
brew install render

"""

"""

git clone https://github.com/render-oss/render-cli
cd render-cli
make deps
deno task run
deno compile

"""

"""

choco install rendercli

"""
</CodeGroup>

In case you run into issues, refer to official [render.com docs](https://docs.render.com/docs/cli).

## Step-2 Create RAG application: 

We provide a command line utility called `ec` in embedchain that inherits the template for `render.com` platform and help you deploy the app. Follow the instructions to create a render.com app using the template provided:
"""
logger.info("## Step-2 Create RAG application:")

pip install embedchain
mkdir my-rag-app
ec create --template=render.com

"""
This `create` command will open a browser window and ask you to login to your render.com account and will generate a directory structure like this:
"""
logger.info("This `create` command will open a browser window and ask you to login to your render.com account and will generate a directory structure like this:")

├── app.py
├── .env
├── render.yaml
├── embedchain.json
└── requirements.txt

"""
Feel free to edit the files as required.
- `app.py`: Contains API app code
- `.env`: Contains environment variables for production
- `render.yaml`: Contains render.com specific configuration for deployment (configure this according to your needs, follow [this](https://docs.render.com/docs/blueprint-spec) for more info)
- `embedchain.json`: Contains embedchain specific configuration for deployment (you don't need to configure this)
- `requirements.txt`: Contains python dependencies for your application

## Step-3: Test app locally

You can run the app locally by simply doing:
"""
logger.info("## Step-3: Test app locally")

pip install -r requirements.txt
ec dev

"""
## Step-4: Deploy to render.com

Before deploying to render.com, you only have to set up one thing. 

In the render.yaml file, make sure to modify the repo key by inserting the URL of your Git repository where your application will be hosted. You can create a repository from [GitHub](https://github.com) or [GitLab](https://gitlab.com/users/sign_in).

After that, you're ready to deploy on render.com.
"""
logger.info("## Step-4: Deploy to render.com")

ec deploy

"""
When you run this, it should open up your render dashboard and you can see the app being deployed. You can find your hosted link over there only.

You can also check the logs, monitor app status etc on their dashboard by running command `render dashboard`.

<img src="/images/fly_io.png" />

## Seeking help?

If you run into issues with deployment, please feel free to reach out to us via any of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Seeking help?")

logger.info("\n\n[DONE]", bright=True)
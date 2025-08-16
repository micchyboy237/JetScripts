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
# Website

This website is built using [Docusaurus 3](https://docusaurus.io/), a modern static website generator.

## Prerequisites

To build and test documentation locally, begin by downloading and installing [Node.js](https://nodejs.org/en/download/), and then installing [Yarn](https://classic.yarnpkg.com/en/).
On Windows, you can install via the npm package manager (npm) which comes bundled with Node.js:
"""
logger.info("# Website")

npm install --global yarn

"""
## Installation
"""
logger.info("## Installation")

pip install pydoc-markdown pyyaml colored
cd website
yarn install

"""
### Install Quarto

`quarto` is used to render notebooks.

Install it [here](https://github.com/quarto-dev/quarto-cli/releases).

> Note: Ensure that your `quarto` version is `1.5.23` or higher.

## Local Development

Navigate to the `website` folder and run:
"""
logger.info("### Install Quarto")

pydoc-markdown
python ./process_notebooks.py render
yarn start

"""
This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.
"""
logger.info("This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.")

logger.info("\n\n[DONE]", bright=True)
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
# Installation

## Create a Virtual Environment (optional)

When installing AgentChat locally, we recommend using a virtual environment for the installation. This will ensure that the dependencies for AgentChat are isolated from the rest of your system.

Create and activate:

Linux/Mac:
"""
logger.info("# Installation")

python3 -m venv .venv
source .venv/bin/activate

"""
Windows command-line:
"""
logger.info("Windows command-line:")

python3 -m venv .venv
.venv\Scripts\activate.bat

"""
To deactivate later, run:
"""
logger.info("To deactivate later, run:")

deactivate

"""


[Install Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) if you have not already.


Create and activate:
"""
logger.info("Create and activate:")

conda create -n autogen python=3.12
conda activate autogen

"""
To deactivate later, run:
"""
logger.info("To deactivate later, run:")

conda deactivate

"""


## Install using pip

Install the `autogen-core` package using pip:
"""
logger.info("## Install using pip")

pip install "autogen-core"

"""

"""

Python 3.10 or later is required.

"""
## Install MLX for Model Client

To use the MLX and Azure MLX models, you need to install the following
extensions:
"""
logger.info("## Install MLX for Model Client")

pip install "autogen-ext[openai]"

"""
If you are using Azure MLX with AAD authentication, you need to install the following:
"""
logger.info("If you are using Azure MLX with AAD authentication, you need to install the following:")

pip install "autogen-ext[azure]"

"""
## Install Docker for Code Execution (Optional)

We recommend using Docker to use {py:class}`~autogen_ext.code_executors.docker.DockerCommandLineCodeExecutor` for execution of model-generated code.
To install Docker, follow the instructions for your operating system on the [Docker website](https://docs.docker.com/get-docker/).

To learn more code execution, see [Command Line Code Executors](./components/command-line-code-executors.ipynb)
and [Code Execution](./design-patterns/code-execution-groupchat.ipynb).
"""
logger.info("## Install Docker for Code Execution (Optional)")

logger.info("\n\n[DONE]", bright=True)
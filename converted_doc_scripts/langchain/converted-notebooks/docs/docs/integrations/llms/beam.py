from jet.logger import logger
from langchain_community.llms.beam import Beam
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Beam

Calls the Beam API wrapper to deploy and make subsequent calls to an instance of the gpt2 LLM in a cloud deployment. Requires installation of the Beam library and registration of Beam Client ID and Client Secret. By calling the wrapper an instance of the model is created and run, with returned text relating to the prompt. Additional calls can then be made by directly calling the Beam API.

[Create an account](https://www.beam.cloud/), if you don't have one already. Grab your API keys from the [dashboard](https://www.beam.cloud/dashboard/settings/api-keys).

Install the Beam CLI
"""
logger.info("# Beam")

# !curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh

"""
Register API Keys and set your beam client id and secret environment variables:
"""
logger.info("Register API Keys and set your beam client id and secret environment variables:")


beam_client_id = "<Your beam client id>"
beam_client_secret = "<Your beam client secret>"

os.environ["BEAM_CLIENT_ID"] = beam_client_id
os.environ["BEAM_CLIENT_SECRET"] = beam_client_secret

# !beam configure --clientId={beam_client_id} --clientSecret={beam_client_secret}

"""
Install the Beam SDK:
"""
logger.info("Install the Beam SDK:")

# %pip install --upgrade --quiet  beam-sdk

"""
**Deploy and call Beam directly from langchain!**

Note that a cold start might take a couple of minutes to return the response, but subsequent calls will be faster!
"""
logger.info("Note that a cold start might take a couple of minutes to return the response, but subsequent calls will be faster!")


llm = Beam(
    model_name="gpt2",
    name="langchain-gpt2-test",
    cpu=8,
    memory="32Gi",
    gpu="A10G",
    python_version="python3.8",
    python_packages=[
        "diffusers[torch]>=0.10",
        "transformers",
        "torch",
        "pillow",
        "accelerate",
        "safetensors",
        "xformers",
    ],
    max_length="50",
    verbose=False,
)

llm._deploy()

response = llm._call("Running machine learning on a remote GPU")

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
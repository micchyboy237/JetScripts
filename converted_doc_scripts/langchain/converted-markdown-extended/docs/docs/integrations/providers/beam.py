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

>[Beam](https://www.beam.cloud/) is a cloud computing platform that allows you to run your code
> on remote servers with GPUs.


## Installation and Setup

- [Create an account](https://www.beam.cloud/)
- Install the Beam CLI with `curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh`
- Register API keys with `beam configure`
- Set environment variables (`BEAM_CLIENT_ID`) and (`BEAM_CLIENT_SECRET`)
- Install the Beam SDK:
"""
logger.info("# Beam")

pip install beam-sdk

"""
## LLMs

See a [usage example](/docs/integrations/llms/beam).

See another example in the [Beam documentation](https://docs.beam.cloud/examples/langchain).
"""
logger.info("## LLMs")


logger.info("\n\n[DONE]", bright=True)
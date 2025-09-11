from jet.logger import logger
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
# Jenkins

[Jenkins](https://www.jenkins.io/) is an open-source automation platform that enables
software teams to streamline their development workflows. It's widely adopted in the
DevOps community as a tool for automating the building, testing, and deployment of
applications through CI/CD pipelines.


## Installation and Setup
"""
logger.info("# Jenkins")

pip install langchain-jenkins

"""
## Tools

See detail on available tools [here](/docs/integrations/tools/jenkins).
"""
logger.info("## Tools")

logger.info("\n\n[DONE]", bright=True)
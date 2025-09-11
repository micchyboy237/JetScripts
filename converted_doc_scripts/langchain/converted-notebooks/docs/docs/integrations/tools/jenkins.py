from jet.logger import logger
from langchain_jenkins import JenkinsAPIWrapper, JenkinsJobRun
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

Tools for interacting with [Jenkins](https://www.jenkins.io/).

## Overview

The `langchain-jenkins` package allows you to execute and control CI/CD pipelines with
Jenkins.

### Setup

Install `langchain-jenkins`:
"""
logger.info("# Jenkins")

# %pip install --upgrade --quiet langchain-jenkins

"""
### Credentials

You'll need to setup or obtain authorization to access Jenkins server.
"""
logger.info("### Credentials")

# import getpass


def _set_env(var: str):
    if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("PASSWORD")

"""
## Instantiation
To disable the SSL Verify, set `os.environ["PYTHONHTTPSVERIFY"] = "0"`
"""
logger.info("## Instantiation")


tools = [
    JenkinsJobRun(
        api_wrapper=JenkinsAPIWrapper(
            jenkins_server="https://example.com",
            username="admin",
            password=os.environ["PASSWORD"],
        )
    )
]

"""
## Invocation
You can now call invoke and pass arguments.

1. Create the Jenkins job
"""
logger.info("## Invocation")

jenkins_job_content = ""
src_file = "job1.xml"
with open(src_file) as fread:
    jenkins_job_content = fread.read()
tools[0].invoke({"job": "job01", "config_xml": jenkins_job_content, "action": "create"})

"""
2. Run the Jenkins Job
"""
logger.info("2. Run the Jenkins Job")

tools[0].invoke({"job": "job01", "parameters": {}, "action": "run"})

"""
3. Get job info
"""
logger.info("3. Get job info")

resp = tools[0].invoke({"job": "job01", "number": 1, "action": "status"})
if not resp["inProgress"]:
    logger.debug(resp["result"])

"""
4. Delete the jenkins job
"""
logger.info("4. Delete the jenkins job")

tools[0].invoke({"job": "job01", "action": "delete"})

"""
## Chaining

TODO.

## API reference

For detailed documentation [API reference](https://python.langchain.com/docs/integrations/tools/jenkins/)
"""
logger.info("## Chaining")

logger.info("\n\n[DONE]", bright=True)
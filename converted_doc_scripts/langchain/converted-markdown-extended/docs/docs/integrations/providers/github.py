from jet.logger import logger
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.document_loaders import GitHubIssuesLoader, GithubFileLoader
from langchain_community.tools.github.tool import GitHubAction
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
# GitHub

>[GitHub](https://github.com/) is a developer platform that allows developers to create,
> store, manage and share their code. It uses `Git` software, providing the
> distributed version control of Git plus access control, bug tracking,
> software feature requests, task management, continuous integration, and wikis for every project.


## Installation and Setup

To access the GitHub API, you need a [personal access token](https://github.com/settings/tokens).


## Document Loader

There are two document loaders available for GitHub.

See a [usage example](/docs/integrations/document_loaders/github).
"""
logger.info("# GitHub")


"""
## Tools/Toolkit

### GitHubToolkit
The `GitHub` toolkit contains tools that enable an LLM agent to interact
with a GitHub repository.

The toolkit is a wrapper for the `PyGitHub` library.
"""
logger.info("## Tools/Toolkit")


"""
Learn more in the [example notebook](/docs/integrations/tools/github).

### GitHubAction

Tool for interacting with the GitHub API.
"""
logger.info("### GitHubAction")


logger.info("\n\n[DONE]", bright=True)
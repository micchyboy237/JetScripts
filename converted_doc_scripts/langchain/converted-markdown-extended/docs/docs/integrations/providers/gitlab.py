from jet.logger import logger
from langchain_community.agent_toolkits.gitlab.toolkit import GitLabToolkit
from langchain_community.tools.gitlab.tool import GitLabAction
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
# GitLab

>[GitLab Inc.](https://about.gitlab.com/) is an open-core company
> that operates `GitLab`, a DevOps software package that can develop,
> secure, and operate software. `GitLab` includes a distributed version
> control based on Git, including features such as access control, bug tracking,
> software feature requests, task management, and wikis for every project,
> as well as snippets.


## Tools/Toolkits

### GitLabToolkit

The `Gitlab` toolkit contains tools that enable an LLM agent to interact with a gitlab repository.

The toolkit is a wrapper for the `python-gitlab` library.

See a [usage example](/docs/integrations/tools/gitlab).
"""
logger.info("# GitLab")


"""
### GitLabAction

Tool for interacting with the GitLab API.
"""
logger.info("### GitLabAction")


logger.info("\n\n[DONE]", bright=True)
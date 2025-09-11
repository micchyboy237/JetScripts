from datetime import datetime
from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.clickup.toolkit import ClickupToolkit
from langchain_community.utilities.clickup import ClickupAPIWrapper
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
# ClickUp Toolkit

>[ClickUp](https://clickup.com/) is an all-in-one productivity platform that provides small and large teams across industries with flexible and customizable work management solutions, tools, and functions. 

>It is a cloud-based project management solution for businesses of all sizes featuring communication and collaboration tools to help achieve organizational goals.
"""
logger.info("# ClickUp Toolkit")

# %pip install -qU langchain-community

# %reload_ext autoreload
# %autoreload 2


"""
## Initializing

### Get Authenticated
1. Create a [ClickUp App](https://help.clickup.com/hc/en-us/articles/6303422883095-Create-your-own-app-with-the-ClickUp-API)
2. Follow [these steps](https://clickup.com/api/developer-portal/authentication/) to get your `client_id` and `client_secret`.
    - *Suggestion: use `https://google.com` as the redirect_uri. This is what we assume in the defaults for this toolkit.*
3. Copy/paste them and run the next cell to get your `code`
"""
logger.info("## Initializing")

oauth_client_id = "ABC..."
oauth_client_secret = "123..."
redirect_uri = "https://google.com"

logger.debug("Click this link, select your workspace, click `Connect Workspace`")
logger.debug(ClickupAPIWrapper.get_access_code_url(oauth_client_id, redirect_uri))

"""
The url should change to something like this https://www.google.com/?code=THISISMYCODERIGHTHERE.

Next, copy/paste the `CODE` (THISISMYCODERIGHTHERE) generated in the URL in the cell below.
"""
logger.info("The url should change to something like this https://www.google.com/?code=THISISMYCODERIGHTHERE.")

code = "THISISMYCODERIGHTHERE"

"""
### Get Access Token
Then, use the code below to get your `access_token`.

*Important*: Each code is a one time code that will expire after use. The `access_token` can be used for a period of time. Make sure to copy paste the `access_token` once you get it!
"""
logger.info("### Get Access Token")

access_token = ClickupAPIWrapper.get_access_token(
    oauth_client_id, oauth_client_secret, code
)

clickup_api_wrapper = ClickupAPIWrapper(access_token=access_token)
toolkit = ClickupToolkit.from_clickup_api_wrapper(clickup_api_wrapper)
logger.debug(
    f"Found team_id: {clickup_api_wrapper.team_id}.\nMost request require the team id, so we store it for you in the toolkit, we assume the first team in your list is the one you want. \nNote: If you know this is the wrong ID, you can pass it at initialization."
)

"""
### Create Agent
"""
logger.info("### Create Agent")

llm = Ollama(temperature=0, ollama_)

agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

"""
## Use an Agent
"""
logger.info("## Use an Agent")

def print_and_run(command):
    logger.debug("\033[94m$ COMMAND\033[0m")
    logger.debug(command)
    logger.debug("\n\033[94m$ AGENT\033[0m")
    response = agent.run(command)
    logger.debug("".join(["-"] * 80))
    return response

"""
### Navigation
You can get the teams, folder and spaces your user has access to
"""
logger.info("### Navigation")

print_and_run("Get all the teams that the user is authorized to access")
print_and_run("Get all the spaces available to the team")
print_and_run("Get all the folders for the team")

"""
### Task Operations
You can get, ask question about tasks and update them
"""
logger.info("### Task Operations")

task_id = "8685mb5fn"

"""
#### Basic attirbute getting and updating
"""
logger.info("#### Basic attirbute getting and updating")

print_and_run(f"Get task with id {task_id}")

previous_description = print_and_run(
    f"What is the description of task with id {task_id}"
)

print_and_run(
    f"For task with id {task_id}, change the description to 'A cool task descriptiont changed by AI!'"
)
print_and_run(f"What is the description of task with id {task_id}")

print_and_run(
    f"For task with id {task_id}, change the description to '{previous_description}'"
)

print_and_run("Change the descrition task 8685mj6cd to 'Look ma no hands'")

"""
#### Advanced Attributes (Assignees)
You can query and update almost every thing about a task!
"""
logger.info("#### Advanced Attributes (Assignees)")

user_id = 81928627

print_and_run(f"What are the assignees of task id {task_id}?")
print_and_run(f"Remove user {user_id} from the assignees of task id {task_id}")
print_and_run(f"What are the assignees of task id {task_id}?")
print_and_run(f"Add user {user_id} from the assignees of task id {task_id}")

"""
### Creation
You can create tasks, lists and folders
"""
logger.info("### Creation")

time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
print_and_run(
    f"Create a task called 'Test Task - {time_str}' with description 'This is a Test'"
)

time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
print_and_run(f"Create a list called Test List - {time_str}")

time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
print_and_run(f"Create a folder called 'Test Folder - {time_str}'")

time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
print_and_run(
    f"Create a list called 'Test List - {time_str}' with content My test list with high priority and status red"
)

"""
## Multi-Step Tasks
"""
logger.info("## Multi-Step Tasks")

print_and_run(
    "Figure out what user ID Rodrigo is, create a task called 'Rod's task', assign it to Rodrigo"
)

logger.info("\n\n[DONE]", bright=True)
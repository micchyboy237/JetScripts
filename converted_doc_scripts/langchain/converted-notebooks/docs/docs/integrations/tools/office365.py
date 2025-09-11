from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits import O365Toolkit
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
# Office365 Toolkit

>[Microsoft 365](https://www.office.com/) is a product family of productivity software, collaboration and cloud-based services owned by `Microsoft`.
>
>Note: `Office 365` was rebranded as `Microsoft 365`.

This notebook walks through connecting LangChain to `Office365` email and calendar.

To use this toolkit, you need to set up your credentials explained in the [Microsoft Graph authentication and authorization overview](https://learn.microsoft.com/en-us/graph/auth/). Once you've received a CLIENT_ID and CLIENT_SECRET, you can input them as environmental variables below.

You can also use the [authentication instructions from here](https://o365.github.io/python-o365/latest/getting_started.html#oauth-setup-pre-requisite).
"""
logger.info("# Office365 Toolkit")

# %pip install --upgrade --quiet  O365
# %pip install --upgrade --quiet  beautifulsoup4  # This is optional but is useful for parsing HTML messages
# %pip install -qU langchain-community

"""
## Assign Environmental Variables

# The toolkit will read the `CLIENT_ID` and `CLIENT_SECRET` environmental variables to authenticate the user so you need to set them here. You will also need to set your `OPENAI_API_KEY` to use the agent later.
"""
logger.info("## Assign Environmental Variables")


"""
## Create the Toolkit and Get Tools

To start, you need to create the toolkit, so you can access its tools later.
"""
logger.info("## Create the Toolkit and Get Tools")


toolkit = O365Toolkit()
tools = toolkit.get_tools()
tools

"""
## Use within an Agent
"""
logger.info("## Use within an Agent")


llm = ChatOllama(temperature=0)
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    verbose=False,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)

agent.run(
    "Create an email draft for me to edit of a letter from the perspective of a sentient parrot"
    " who is looking to collaborate on some research with her"
    " estranged friend, a cat. Under no circumstances may you send the message, however."
)

agent.run(
    "Could you search in my drafts folder and let me know if any of them are about collaboration?"
)

agent.run(
    "Can you schedule a 30 minute meeting with a sentient parrot to discuss research collaborations on October 3, 2023 at 2 pm Easter Time?"
)

agent.run(
    "Can you tell me if I have any events on October 3, 2023 in Eastern Time, and if so, tell me if any of them are with a sentient parrot?"
)

logger.info("\n\n[DONE]", bright=True)

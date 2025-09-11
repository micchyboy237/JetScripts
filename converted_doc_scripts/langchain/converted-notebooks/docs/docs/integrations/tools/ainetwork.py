from ain.account import Account
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.agent_toolkits.ainetwork.toolkit import AINetworkToolkit
from langgraph.prebuilt import create_react_agent
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
# AINetwork Toolkit

>[AI Network](https://www.ainetwork.ai/build-on-ain) is a layer 1 blockchain designed to accommodate large-scale AI models, utilizing a decentralized GPU network powered by the [$AIN token](https://www.ainetwork.ai/token), enriching AI-driven `NFTs` (`AINFTs`).
>
>The `AINetwork Toolkit` is a set of tools for interacting with the [AINetwork Blockchain](https://www.ainetwork.ai/public/whitepaper.pdf). These tools allow you to transfer `AIN`, read and write values, create apps, and set permissions for specific paths within the blockchain database.

## Installing dependencies

Before using the AINetwork Toolkit, you need to install the ain-py package. You can install it with pip:
"""
logger.info("# AINetwork Toolkit")

# %pip install --upgrade --quiet  ain-py langchain-community

"""
## Set environmental variables

You need to set the `AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY` environmental variable to your AIN Blockchain Account Private Key.
"""
logger.info("## Set environmental variables")


os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"] = ""

"""
### Get AIN Blockchain private key
"""
logger.info("### Get AIN Blockchain private key")



if os.environ.get("AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY", None):
    account = Account(os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"])
else:
    account = Account.create()
    os.environ["AIN_BLOCKCHAIN_ACCOUNT_PRIVATE_KEY"] = account.private_key
    logger.debug(
        f"""
address: {account.address}
private_key: {account.private_key}
"""
    )

"""
## Initialize the AINetwork Toolkit

You can initialize the AINetwork Toolkit like this:
"""
logger.info("## Initialize the AINetwork Toolkit")


toolkit = AINetworkToolkit()
tools = toolkit.get_tools()
address = tools[0].interface.wallet.defaultAccount.address

"""
## Initialize the Agent with the AINetwork Toolkit

You can initialize the agent with the AINetwork Toolkit like this:
"""
logger.info("## Initialize the Agent with the AINetwork Toolkit")


llm = ChatOllama(model="llama3.2")
agent = create_react_agent(model=llm, tools=tools)

"""
## Example Usage

Here are some examples of how you can use the agent with the AINetwork Toolkit:

### Define App name to test
"""
logger.info("## Example Usage")

appName = f"langchain_demo_{address.lower()}"

"""
### Create an app in the AINetwork Blockchain database
"""
logger.info("### Create an app in the AINetwork Blockchain database")

logger.debug(
    agent.invoke(
        {
            "messages": f"Create an app in the AINetwork Blockchain database with the name {appName}"
        }
    )
)

"""
### Set a value at a given path in the AINetwork Blockchain database
"""
logger.info("### Set a value at a given path in the AINetwork Blockchain database")

logger.debug(
    agent.run(f"Set the value {{1: 2, '34': 56}} at the path /apps/{appName}/object .")
)

"""
### Set permissions for a path in the AINetwork Blockchain database
"""
logger.info("### Set permissions for a path in the AINetwork Blockchain database")

logger.debug(
    agent.run(
        f"Set the write permissions for the path /apps/{appName}/user/$from with the"
        " eval string auth.addr===$from ."
    )
)

"""
### Retrieve the permissions for a path in the AINetwork Blockchain database
"""
logger.info("### Retrieve the permissions for a path in the AINetwork Blockchain database")

logger.debug(agent.run(f"Retrieve the permissions for the path /apps/{appName}."))

"""
### Get AIN from faucet
"""
logger.info("### Get AIN from faucet")

# !curl http://faucet.ainetwork.ai/api/test/{address}/

"""
### Get AIN Balance
"""
logger.info("### Get AIN Balance")

logger.debug(agent.run(f"Check AIN balance of {address}"))

"""
### Transfer AIN
"""
logger.info("### Transfer AIN")

logger.debug(
    agent.run(
        "Transfer 100 AIN to the address 0x19937b227b1b13f29e7ab18676a89ea3bdea9c5b"
    )
)

logger.info("\n\n[DONE]", bright=True)
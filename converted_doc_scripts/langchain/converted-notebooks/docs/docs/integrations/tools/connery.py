from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.connery import ConneryToolkit
from langchain_community.tools.connery import ConneryService
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
# Connery Toolkit and Tools

Using the Connery toolkit and tools, you can integrate Connery Actions into your LangChain agent.

## What is Connery?

Connery is an open-source plugin infrastructure for AI.

With Connery, you can easily create a custom plugin with a set of actions and seamlessly integrate them into your LangChain agent.
Connery will take care of critical aspects such as runtime, authorization, secret management, access management, audit logs, and other vital features.

Furthermore, Connery, supported by our community, provides a diverse collection of ready-to-use open-source plugins for added convenience.

Learn more about Connery:

- GitHub: https://github.com/connery-io/connery
- Documentation: https://docs.connery.io

## Setup

### Installation

You need to install the `langchain_community` package to use the Connery tools.
"""
logger.info("# Connery Toolkit and Tools")

# %pip install -qU langchain-community

"""
### Credentials

To use Connery Actions in your LangChain agent, you need to do some preparation:

1. Set up the Connery runner using the [Quickstart](https://docs.connery.io/docs/runner/quick-start/) guide.
2. Install all the plugins with the actions you want to use in your agent.
3. Set environment variables `CONNERY_RUNNER_URL` and `CONNERY_RUNNER_API_KEY` so the toolkit can communicate with the Connery Runner.
"""
logger.info("### Credentials")

# import getpass

for key in ["CONNERY_RUNNER_URL", "CONNERY_RUNNER_API_KEY"]:
    if key not in os.environ:
#         os.environ[key] = getpass.getpass(f"Please enter the value for {key}: ")

"""
## Toolkit

In the example below, we create an agent that uses two Connery Actions to summarize a public webpage and send the summary by email:

1. **Summarize public webpage** action from the [Summarization](https://github.com/connery-io/summarization-plugin) plugin.
2. **Send email** action from the [Gmail](https://github.com/connery-io/gmail) plugin.

You can see a LangSmith trace of this example [here](https://smith.langchain.com/public/4af5385a-afe9-46f6-8a53-57fe2d63c5bc/r).
"""
logger.info("## Toolkit")



os.environ["CONNERY_RUNNER_URL"] = ""
os.environ["CONNERY_RUNNER_API_KEY"] = ""

# os.environ["OPENAI_API_KEY"] = ""

recepient_email = "test@example.com"

connery_service = ConneryService()
connery_toolkit = ConneryToolkit.create_instance(connery_service)

llm = ChatOllama(model="llama3.2")
agent = initialize_agent(
    connery_toolkit.get_tools(), llm, AgentType.OPENAI_FUNCTIONS, verbose=True
)
result = agent.run(
    f"""Make a short summary of the webpage http://www.paulgraham.com/vb.html in three sentences
and send it to {recepient_email}. Include the link to the webpage into the body of the email."""
)
logger.debug(result)

"""
NOTE: Connery Action is a structured tool, so you can only use it in the agents supporting structured tools.

## Tool
"""
logger.info("## Tool")



os.environ["CONNERY_RUNNER_URL"] = ""
os.environ["CONNERY_RUNNER_API_KEY"] = ""

# os.environ["OPENAI_API_KEY"] = ""

recepient_email = "test@example.com"

connery_service = ConneryService()
send_email_action = connery_service.get_action("CABC80BB79C15067CA983495324AE709")

"""
Run the action manually.
"""
logger.info("Run the action manually.")

manual_run_result = send_email_action.run(
    {
        "recipient": recepient_email,
        "subject": "Test email",
        "body": "This is a test email sent from Connery.",
    }
)
logger.debug(manual_run_result)

"""
Run the action using the Ollama Functions agent.

You can see a LangSmith trace of this example [here](https://smith.langchain.com/public/a37d216f-c121-46da-a428-0e09dc19b1dc/r).
"""
logger.info("Run the action using the Ollama Functions agent.")

llm = ChatOllama(model="llama3.2")
agent = initialize_agent(
    [send_email_action], llm, AgentType.OPENAI_FUNCTIONS, verbose=True
)
agent_run_result = agent.run(
    f"Send an email to the {recepient_email} and say that I will be late for the meeting."
)
logger.debug(agent_run_result)

"""
NOTE: Connery Action is a structured tool, so you can only use it in the agents supporting structured tools.

## API reference

For detailed documentation of all Connery features and configurations head to the API reference:

- Toolkit: https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.connery.toolkit.ConneryToolkit.html
- Tool: https://python.langchain.com/api_reference/community/tools/langchain_community.tools.connery.service.ConneryService.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
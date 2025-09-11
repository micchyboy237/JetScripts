from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langgraph.prebuilt import create_react_agent
from stripe_agent_toolkit.crewai.toolkit import StripeAgentToolkit
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
---
sidebar_label: Stripe
---

# StripeAgentToolkit

This notebook provides a quick overview for getting started with Stripe's agent toolkit.

You can read more about `StripeAgentToolkit` in [Stripe's launch blog](https://stripe.dev/blog/adding-payments-to-your-agentic-workflows) or on the project's [PyPi page](https://pypi.org/project/stripe-agent-toolkit/).

## Overview

### Integration details

| Class | Package | Serializable | [JS Support](https://github.com/stripe/agent-toolkit?tab=readme-ov-file#typescript) |  Package latest |
| :--- | :--- | :---: | :---: | :---: |
| StripeAgentToolkit | [stripe-agent-toolkit](https://pypi.org/project/stripe-agent-toolkit) | ❌ | ✅ |  ![PyPI - Version](https://img.shields.io/pypi/v/stripe-agent-toolkit?style=flat-square&label=%20) |


## Setup

This externally-managed package is hosted out of the `stripe-agent-toolkit` project, which is managed by Stripe's team.

You can install it, along with langgraph for the following examples, with `pip`:
"""
logger.info("# StripeAgentToolkit")

# %pip install --quiet -U langgraph stripe-agent-toolkit

"""
### Credentials

In addition to installing the package, you will need to configure the integration with your Stripe account's secret key, which is available in your [Stripe Dashboard](https://dashboard.stripe.com/account/apikeys).
"""
logger.info("### Credentials")

# import getpass

if not os.environ.get("STRIPE_SECRET_KEY"):
#     os.environ["STRIPE_SECRET_KEY"] = getpass.getpass("STRIPE API key:\n")

"""
It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:
"""
logger.info("It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability:")



"""
## Instantiation

Here we show how to create an instance of the Stripe Toolkit
"""
logger.info("## Instantiation")


stripe_agent_toolkit = StripeAgentToolkit(
    secret_key=os.getenv("STRIPE_SECRET_KEY"),
    configuration={
        "actions": {
            "payment_links": {
                "create": True,
            },
        }
    },
)

"""
## Agent

Here's how to use the toolkit to create a basic agent in langgraph:
"""
logger.info("## Agent")


llm = ChatOllama(
    model="llama3.2",
)

langgraph_agent_executor = create_react_agent(llm, stripe_agent_toolkit.get_tools())

input_state = {
    "messages": """
        Create a payment link for a new product called 'test' with a price
        of $100. Come up with a funny description about buy bots,
        maybe a haiku.
    """,
}

output_state = langgraph_agent_executor.invoke(input_state)

logger.debug(output_state["messages"][-1].content)

logger.info("\n\n[DONE]", bright=True)
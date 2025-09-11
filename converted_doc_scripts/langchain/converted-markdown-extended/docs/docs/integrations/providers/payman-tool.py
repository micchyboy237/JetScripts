from jet.logger import logger
from langchain_payman_tool.tool import PaymanAI
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
# PaymanAI

PaymanAI provides functionality to send and receive payments (fiat and crypto) on behalf of an AI Agent. To get started:

1. **Sign up** at app.paymanai.com to create an AI Agent and obtain your **API Key**.
2. **Set** environment variables (`PAYMAN_API_SECRET` for your API Key, `PAYMAN_ENVIRONMENT` for sandbox or production).

This notebook gives a quick overview of integrating PaymanAI into LangChain as a tool. For complete reference, see the API documentation.

## Overview

The PaymanAI integration is part of the `langchain-community` (or your custom) package. It allows you to:

- Send payments (`send_payment`) to crypto addresses or bank accounts.
- Search for payees (`search_payees`).
- Add new payees (`add_payee`).
- Request money from customers with a hosted checkout link (`ask_for_money`).
- Check agent or customer balances (`get_balance`).

These can be wrapped as **LangChain Tools** for an LLM-based agent to call them automatically.

### Integration details

| Class | Package | Serializable | JS support | Package latest |
| :--- | :--- | :---: | :---: | :--- |
| PaymanAI | `langchain-payman-tool` | ❌ | ❌ | [PyPI Version] |

If you're simply calling the PaymanAI SDK, you can do it directly or via the **Tool** interface in LangChain.

## Setup

1. **Install** the PaymanAI tool package:
"""
logger.info("# PaymanAI")

pip install langchain-payman-tool

"""
2. **Install** the PaymanAI SDK:
"""
logger.info("2. **Install** the PaymanAI SDK:")

pip install paymanai

"""
3. **Set** environment variables:
"""
logger.info("3. **Set** environment variables:")

export PAYMAN_API_SECRET="YOUR_SECRET_KEY"
export PAYMAN_ENVIRONMENT="sandbox"

"""
Your `PAYMAN_API_SECRET` should be the secret key from app.paymanai.com. The `PAYMAN_ENVIRONMENT` can be `sandbox` or `production` depending on your usage.

## Instantiation

Here is an example of instantiating a PaymanAI tool. If you have multiple Payman methods, you can create multiple tools.
"""
logger.info("## Instantiation")


tool = PaymanAI(
    name="send_payment",
    description="Send a payment to a specified payee.",
)

"""
## Invocation

### Invoke directly with args

You can call `tool.invoke(...)` and pass a dictionary matching the tool's expected fields. For example:
"""
logger.info("## Invocation")

response = tool.invoke({
    "amount_decimal": 10.00,
    "payment_destination_id": "abc123",
    "customer_id": "cust_001",
    "memo": "Payment for invoice #XYZ"
})

"""
### Invoke with ToolCall

When used inside an AI workflow, the LLM might produce a `ToolCall` dict. You can simulate it as follows:
"""
logger.info("### Invoke with ToolCall")

model_generated_tool_call = {
    "args": {
        "amount_decimal": 10.00,
        "payment_destination_id": "abc123"
    },
    "id": "1",
    "name": tool.name,
    "type": "tool_call",
}
tool.invoke(model_generated_tool_call)

"""
## Using the Tool in a Chain or Agent

You can bind a PaymanAI tool to a LangChain agent or chain that supports tool-calling.

## Quick Start Summary

1. **Sign up** at app.paymanai.com to get your **API Key**.
2. **Install** dependencies:
   ```bash
#    pip install paymanai langchain-payman-tool
   ```
3. **Export** environment variables:
   ```bash
   export PAYMAN_API_SECRET="YOUR_SECRET_KEY"
   export PAYMAN_ENVIRONMENT="sandbox"
   ```
4. **Instantiate** a PaymanAI tool, passing your desired name/description.
5. **Call** the tool with `.invoke(...)` or integrate it into a chain or agent.
"""
logger.info("## Using the Tool in a Chain or Agent")

logger.info("\n\n[DONE]", bright=True)
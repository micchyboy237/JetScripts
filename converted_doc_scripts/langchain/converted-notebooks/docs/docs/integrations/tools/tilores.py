from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from tilores import TiloresAPI
from tilores_langchain import TiloresTools
import ChatModelTabs from "@theme/ChatModelTabs";
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
sidebar_label: Tilores
---

# Tilores

This notebook covers how to get started with the [Tilores](/docs/integrations/providers/tilores) tools.
For a more complex example you can checkout our [customer insights chatbot example](https://github.com/tilotech/identity-rag-customer-insights-chatbot).

## Overview

### Integration details

| Class | Package | Serializable | JS support |  Package latest |
| :--- | :--- | :---: | :---: | :---: |
| TiloresTools | [tilores-langchain](https://pypi.org/project/tilores-langchain/) | ❌ | ❌ |  ![PyPI - Version](https://img.shields.io/pypi/v/tilores-langchain?style=flat-square&label=%20) |

## Setup

The integration requires the following packages:
"""
logger.info("# Tilores")

# %pip install --quiet -U tilores-langchain langchain

"""
### Credentials

To access Tilores, you need to [create and configure an instance](https://app.tilores.io). If you prefer to test out Tilores first, you can use the [read-only demo credentials](https://github.com/tilotech/identity-rag-customer-insights-chatbot?tab=readme-ov-file#1-configure-customer-data-access).
"""
logger.info("### Credentials")


os.environ["TILORES_API_URL"] = "<api-url>"
os.environ["TILORES_TOKEN_URL"] = "<token-url>"
os.environ["TILORES_CLIENT_ID"] = "<client-id>"
os.environ["TILORES_CLIENT_SECRET"] = "<client-secret>"

"""
## Instantiation

Here we show how to instantiate an instance of the Tilores tools:
"""
logger.info("## Instantiation")


tilores = TiloresAPI.from_environ()
tilores_tools = TiloresTools(tilores)
search_tool = tilores_tools.search_tool()
edge_tool = tilores_tools.edge_tool()

"""
## Invocation

The parameters for the `tilores_search` tool are dependent on the [configured schema](https://docs.tilotech.io/tilores/schema/) within Tilores. The following examples will use the schema for the demo instance with generated data.

### [Invoke directly with args](/docs/concepts/tools)

The following example searches for a person called Sophie Müller in Berlin. The Tilores data contains multiple such persons and returns their known email addresses and phone numbers.
"""
logger.info("## Invocation")

result = search_tool.invoke(
    {
        "searchParams": {
            "name": "Sophie Müller",
            "city": "Berlin",
        },
        "recordFieldsToQuery": {
            "email": True,
            "phone": True,
        },
    }
)
logger.debug("Number of entities:", len(result["data"]["search"]["entities"]))
for entity in result["data"]["search"]["entities"]:
    logger.debug("Number of records:", len(entity["records"]))
    logger.debug(
        "Email Addresses:",
        [record["email"] for record in entity["records"] if record.get("email")],
    )
    logger.debug(
        "Phone Numbers:",
        [record["phone"] for record in entity["records"] if record.get("phone")],
    )

"""
If we're interested how the records from the first entity are related, we can use the edge_tool. Note that the Tilores entity resolution engine figured out the relation between those records automatically. Please refer to the [edge documentation](https://docs.tilotech.io/tilores/rules/#edges) for more details.
"""
logger.info("If we're interested how the records from the first entity are related, we can use the edge_tool. Note that the Tilores entity resolution engine figured out the relation between those records automatically. Please refer to the [edge documentation](https://docs.tilotech.io/tilores/rules/#edges) for more details.")

edge_result = edge_tool.invoke(
    {"entityID": result["data"]["search"]["entities"][0]["id"]}
)
edges = edge_result["data"]["entity"]["entity"]["edges"]
logger.debug("Number of edges:", len(edges))
logger.debug("Edges:", edges)

"""
### [Invoke with ToolCall](/docs/concepts/tools)

We can also invoke the tool with a model-generated ToolCall, in which case a ToolMessage will be returned:
"""
logger.info("### [Invoke with ToolCall](/docs/concepts/tools)")

model_generated_tool_call = {
    "args": {
        "searchParams": {
            "name": "Sophie Müller",
            "city": "Berlin",
        },
        "recordFieldsToQuery": {
            "email": True,
            "phone": True,
        },
    },
    "id": "1",
    "name": search_tool.name,
    "type": "tool_call",
}
search_tool.invoke(model_generated_tool_call)

"""
## Chaining

We can use our tool in a chain by first binding it to a [tool-calling model](/docs/how_to/tool_calling/) and then calling it:


<ChatModelTabs customVarName="llm" />
"""
logger.info("## Chaining")


llm = init_chat_model(model="llama3.2", model_provider="ollama")


prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

llm_with_tools = llm.bind_tools([search_tool], tool_choice=search_tool.name)

llm_chain = prompt | llm_with_tools


@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = search_tool.batch(ai_msg.tool_calls, config=config)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


tool_chain.invoke("Tell me the email addresses from Sophie Müller from Berlin.")

"""
## API reference

For detailed documentation of all Tilores features and configurations head to the official documentation: https://docs.tilotech.io/tilores/
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
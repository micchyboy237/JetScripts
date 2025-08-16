from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.retrieve_utils import TEXT_FORMATS
from jet.logger import CustomLogger
import autogen
import json
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using RetrieveChat Powered by MongoDB Atlas for Retrieve Augmented Code Generation and Question Answering

AutoGen offers conversable agents powered by LLM, tool or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

RetrieveChat is a conversational system for retrieval-augmented code generation and question answering. In this notebook, we demonstrate how to utilize RetrieveChat to generate code and answer questions based on customized documentations that are not present in the LLM's training dataset. RetrieveChat uses the `AssistantAgent` and `RetrieveUserProxyAgent`, which is similar to the usage of `AssistantAgent` and `UserProxyAgent` in other notebooks (e.g., [Automated Task Solving with Code Generation, Execution & Debugging](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_auto_feedback_from_code_execution.ipynb)). Essentially, `RetrieveUserProxyAgent` implement a different auto-reply mechanism corresponding to the RetrieveChat prompts.

## Table of Contents
We'll demonstrate six examples of using RetrieveChat for code generation and question answering:

- [Example 1: Generate code based off docstrings w/o human feedback](#example-1)

````{=mdx}
:::info Requirements
Some extra dependencies are needed for this notebook, which can be installed via pip:

```bash
pip install autogen[retrievechat-mongodb] flaml[automl]
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````

Ensure you have a MongoDB Atlas instance with Cluster Tier >= M10. Read more on Cluster support [here](https://www.mongodb.com/docs/atlas/atlas-search/manage-indexes/#create-and-manage-fts-indexes)

## Set your API Endpoint
"""
logger.info("# Using RetrieveChat Powered by MongoDB Atlas for Retrieve Augmented Code Generation and Question Answering")




# config_list = [{"model": "gpt-3.5-turbo-0125", "api_key": os.environ["OPENAI_API_KEY"], "api_type": "openai"}]
assert len(config_list) > 0
logger.debug("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````

## Construct agents for RetrieveChat

We start by initializing the `AssistantAgent` and `RetrieveUserProxyAgent`. The system message needs to be set to "You are a helpful assistant." for AssistantAgent. The detailed instructions are given in the user message. Later we will use the `RetrieveUserProxyAgent.message_generator` to combine the instructions and a retrieval augmented generation task for an initial prompt to be sent to the LLM assistant.
"""
logger.info("## Construct agents for RetrieveChat")

logger.debug("Accepted file formats for `docs_path`:")
logger.debug(TEXT_FORMATS)

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
        ],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "vector_db": "mongodb",  # MongoDB Atlas database
        "collection_name": "demo_collection",
        "db_config": {
            "connection_string": os.environ["MONGODB_URI"],  # MongoDB Atlas connection string
            "database_name": "test_db",  # MongoDB Atlas database
            "index_name": "vector_index",
            "wait_until_index_ready": 120.0,  # Setting to wait 120 seconds or until index is constructed before querying
            "wait_until_document_ready": 120.0,  # Setting to wait 120 seconds or until document is properly indexed after insertion/update
        },
        "get_or_create": True,  # set to False if you don't want to reuse an existing collection
        "overwrite": False,  # set to True if you want to overwrite an existing collection, each overwrite will force a index creation and reupload of documents
    },
    code_execution_config=False,  # set to False if you don't want to execute the code
)

"""
### Example 1

[Back to top](#table-of-contents)

Use RetrieveChat to help generate sample code and automatically run the code and fix errors if there is any.

Problem: Which API should I use if I want to use FLAML for a classification task and I want to train the model in 30 seconds. Use spark to parallel the training. Force cancel jobs if time limit is reached.
"""
logger.info("### Example 1")

assistant.reset()

code_problem = "How can I use FLAML to perform a classification task and use spark to do parallel training. Train 30 seconds and force cancel jobs if time limit is reached."
chat_result = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=code_problem)

logger.info("\n\n[DONE]", bright=True)
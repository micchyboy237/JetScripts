from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.retrieve_utils import TEXT_FORMATS
from jet.logger import CustomLogger
from sentence_transformers import SentenceTransformer
import autogen
import chromadb
import json
import os
import psycopg

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using RetrieveChat Powered by PGVector for Retrieve Augmented Code Generation and Question Answering

AutoGen offers conversable agents powered by LLM, tool or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation.
Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

RetrieveChat is a conversational system for retrieval-augmented code generation and question answering. In this notebook, we demonstrate how to utilize RetrieveChat to generate code and answer questions based on customized documentations that are not present in the LLM's training dataset. RetrieveChat uses the `AssistantAgent` and `RetrieveUserProxyAgent`, which is similar to the usage of `AssistantAgent` and `UserProxyAgent` in other notebooks (e.g., [Automated Task Solving with Code Generation, Execution & Debugging](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_auto_feedback_from_code_execution.ipynb)). Essentially, `RetrieveUserProxyAgent` implement a different auto-reply mechanism corresponding to the RetrieveChat prompts.

## Table of Contents
We'll demonstrate six examples of using RetrieveChat for code generation and question answering:

- [Example 1: Generate code based off docstrings w/o human feedback](#example-1)
- [Example 2: Answer a question based off docstrings w/o human feedback](#example-2)


````{=mdx}
:::info Requirements
Some extra dependencies are needed for this notebook, which can be installed via pip:

```bash
pip install autogen[retrievechat-pgvector] flaml[automl]
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````

Ensure you have a PGVector instance. 

If not, a test version can quickly be deployed using Docker.

`docker-compose.yml`
```yml
version: '3.9'

services:
  pgvector:
    image: pgvector/pgvector:pg16
    shm_size: 128mb
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: <postgres-user>
      POSTGRES_PASSWORD: <postgres-password>
      POSTGRES_DB: <postgres-database>
    volumes:
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
```

Create `init.sql` file
```SQL
CREATE EXTENSION IF NOT EXISTS vector;
```

## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("# Using RetrieveChat Powered by PGVector for Retrieve Augmented Code Generation and Question Answering")





config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    file_location=".",
)
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
    system_message="You are a helpful assistant. You must always reply with some form of text.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)


sentence_transformer_ef = SentenceTransformer("all-distilroberta-v1").encode

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
        ],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "vector_db": "pgvector",  # PGVector database
        "collection_name": "flaml_collection",
        "db_config": {
            "connection_string": "postgresql://postgres:postgres@localhost:5432/postgres",  # Optional - connect to an external vector database
        },
        "get_or_create": True,  # set to False if you don't want to reuse an existing collection
        "overwrite": True,  # set to True if you want to overwrite an existing collection
        "embedding_function": sentence_transformer_ef,  # If left out SentenceTransformer("all-MiniLM-L6-v2").encode will be used
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

code_problem = "How can I use FLAML to perform a classification task and use spark to do parallel training. Train for 30 seconds and force cancel jobs if time limit is reached."
chat_result = ragproxyagent.initiate_chat(
    assistant, message=ragproxyagent.message_generator, problem=code_problem, search_string="spark"
)

"""
### Example 2

[Back to top](#table-of-contents)

Use RetrieveChat to answer a question that is not related to code generation.

Problem: Who is the author of FLAML?
"""
logger.info("### Example 2")

assistant.reset()

conn = psycopg.connect(conninfo="postgresql://postgres:postgres@localhost:5432/postgres", autocommit=True)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
            os.path.join(os.path.abspath(""), "..", "website", "docs"),
        ],
        "custom_text_types": ["non-existent-type"],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "vector_db": "pgvector",  # PGVector database
        "collection_name": "flaml_collection",
        "db_config": {
            "conn": conn,  # Optional - conn object to connect to database
        },
        "get_or_create": True,  # set to False if you don't want to reuse an existing collection
        "overwrite": True,  # set to True if you want to overwrite an existing collection
    },
    code_execution_config=False,  # set to False if you don't want to execute the code
)

qa_problem = "Who is the author of FLAML?"
chat_result = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=qa_problem)

logger.info("\n\n[DONE]", bright=True)
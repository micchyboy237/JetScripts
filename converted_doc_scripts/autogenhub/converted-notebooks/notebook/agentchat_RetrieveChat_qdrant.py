from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.retrieve_utils import TEXT_FORMATS
from jet.logger import CustomLogger
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import autogen
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using RetrieveChat with Qdrant for Retrieve Augmented Code Generation and Question Answering

[Qdrant](https://qdrant.tech/) is a high-performance vector search engine/database.

This notebook demonstrates the usage of Qdrant for RAG, based on [agentchat_RetrieveChat.ipynb](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_RetrieveChat.ipynb).


RetrieveChat is a conversational system for retrieve augmented code generation and question answering. In this notebook, we demonstrate how to utilize RetrieveChat to generate code and answer questions based on customized documentations that are not present in the LLM's training dataset. RetrieveChat uses the `AssistantAgent` and `RetrieveUserProxyAgent`, which is similar to the usage of `AssistantAgent` and `UserProxyAgent` in other notebooks (e.g., [Automated Task Solving with Code Generation, Execution & Debugging](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_auto_feedback_from_code_execution.ipynb)).

We'll demonstrate usage of RetrieveChat with Qdrant for code generation and question answering w/ human feedback.

````{=mdx}
:::info Requirements
Some extra dependencies are needed for this notebook, which can be installed via pip:

```bash
pip install "autogen[retrievechat-qdrant]" "flaml[automl]"
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````
"""
logger.info("# Using RetrieveChat with Qdrant for Retrieve Augmented Code Generation and Question Answering")

# %pip install "autogen[retrievechat-qdrant]" "flaml[automl]" -q

"""
## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("## Set your API Endpoint")




config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

assert len(config_list) > 0
logger.debug("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````
"""
logger.info("Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).")

logger.debug("Accepted file formats for `docs_path`:")
logger.debug(TEXT_FORMATS)

"""
## Construct agents for RetrieveChat

We start by initializing the `AssistantAgent` and `RetrieveUserProxyAgent`. The system message needs to be set to "You are a helpful assistant." for AssistantAgent. The detailed instructions are given in the user message. Later we will use the `RetrieveUserProxyAgent.generate_init_prompt` to combine the instructions and a retrieval augmented generation task for an initial prompt to be sent to the LLM assistant.

### You can find the list of all the embedding models supported by Qdrant [here](https://qdrant.github.io/fastembed/examples/Supported_Models/).
"""
logger.info("## Construct agents for RetrieveChat")

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

sentence_transformer_ef = SentenceTransformer("all-distilroberta-v1").encode
client = QdrantClient(":memory:")

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/autogenhub/flaml/main/README.md",
            "https://raw.githubusercontent.com/autogenhub/FLAML/main/website/docs/Research.md",
        ],  # change this to your own path, such as https://raw.githubusercontent.com/autogenhub/autogen/main/README.md
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "db_config": {"client": client},
        "vector_db": "qdrant",  # qdrant database
        "get_or_create": True,  # set to False if you don't want to reuse an existing collection
        "overwrite": True,  # set to True if you want to overwrite an existing collection
        "embedding_function": sentence_transformer_ef,  # If left out fastembed "BAAI/bge-small-en-v1.5" will be used
    },
    code_execution_config=False,
)

"""
<a id="example-1"></a>
### Example 1

[back to top](#toc)

Use RetrieveChat to answer a question and ask for human-in-loop feedbacks.

Problem: Is there a function named `tune_automl` in FLAML?
"""
logger.info("### Example 1")

assistant.reset()

qa_problem = "Is there a function called tune_automl?"
chat_results = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=qa_problem)

"""
<a id="example-2"></a>
### Example 2

[back to top](#toc)

Use RetrieveChat to answer a question that is not related to code generation.

Problem: Who is the author of FLAML?
"""
logger.info("### Example 2")

assistant.reset()

qa_problem = "Who is the author of FLAML?"
chat_results = ragproxyagent.initiate_chat(assistant, message=ragproxyagent.message_generator, problem=qa_problem)

logger.info("\n\n[DONE]", bright=True)
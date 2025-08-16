from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
import logging
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Chat with Ollama Assistant using function call in AutoGen: OSS Insights for Advanced GitHub Data Analysis

This Jupyter Notebook demonstrates how to leverage OSS Insight (Open Source Software Insight) for advanced GitHub data analysis by defining `Function calls` in AutoGen for the Ollama Assistant. 

The notebook is structured into four main sections:

1. Function Schema and Implementation
2. Defining an Ollama Assistant Agent in AutoGen
3. Fetching GitHub Insight Data using Function Call

## Requirements

AutoGen requires `Python>=3.8`. To run this notebook example, please install:
````{=mdx}
:::info Requirements
Install `pyautogen`:
```bash
pip install pyautogen
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````
"""
logger.info("# Chat with Ollama Assistant using function call in AutoGen: OSS Insights for Advanced GitHub Data Analysis")

# %%capture --no-stderr

"""
## Function Schema and Implementation

This section provides the function schema definition and their implementation details. These functions are tailored to fetch and process data from GitHub, utilizing OSS Insight's capabilities.
"""
logger.info("## Function Schema and Implementation")



logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

ossinsight_api_schema = {
    "name": "ossinsight_data_api",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "Enter your GitHub data question in the form of a clear and specific question to ensure the returned data is accurate and valuable. "
                    "For optimal results, specify the desired format for the data table in your request."
                ),
            }
        },
        "required": ["question"],
    },
    "description": "This is an API endpoint allowing users (analysts) to input question about GitHub in text format to retrieve the related and structured data.",
}


def get_ossinsight(question):
    """
    [Mock] Retrieve the top 10 developers with the most followers on GitHub.
    """
    report_components = [
        f"Question: {question}",
        "SQL: SELECT `login` AS `user_login`, `followers` AS `followers` FROM `github_users` ORDER BY `followers` DESC LIMIT 10",
        """Results:
  {'followers': 166730, 'user_login': 'torvalds'}
  {'followers': 86239, 'user_login': 'yyx990803'}
  {'followers': 77611, 'user_login': 'gaearon'}
  {'followers': 72668, 'user_login': 'ruanyf'}
  {'followers': 65415, 'user_login': 'JakeWharton'}
  {'followers': 60972, 'user_login': 'peng-zhihui'}
  {'followers': 58172, 'user_login': 'bradtraversy'}
  {'followers': 52143, 'user_login': 'gustavoguanabara'}
  {'followers': 51542, 'user_login': 'sindresorhus'}
  {'followers': 49621, 'user_login': 'tj'}""",
    ]
    return "\n" + "\n\n".join(report_components) + "\n"

"""
## Defining an Ollama Assistant Agent in AutoGen

Here, we explore how to define an Ollama Assistant Agent within the AutoGen. This includes setting up the agent to make use of the previously defined function calls for data retrieval and analysis.
"""
logger.info("## Defining an Ollama Assistant Agent in AutoGen")

assistant_id = os.environ.get("ASSISTANT_ID", None)
config_list = config_list_from_json("OAI_CONFIG_LIST")
llm_config = {
    "config_list": config_list,
}
assistant_config = {
    "assistant_id": assistant_id,
    "tools": [
        {
            "type": "function",
            "function": ossinsight_api_schema,
        }
    ],
}

oss_analyst = GPTAssistantAgent(
    name="OSS Analyst",
    instructions=(
        "Hello, Open Source Project Analyst. You'll conduct comprehensive evaluations of open source projects or organizations on the GitHub platform, "
        "analyzing project trajectories, contributor engagements, open source trends, and other vital parameters. "
        "Please carefully read the context of the conversation to identify the current analysis question or problem that needs addressing."
    ),
    llm_config=llm_config,
    assistant_config=assistant_config,
    verbose=True,
)
oss_analyst.register_function(
    function_map={
        "ossinsight_data_api": get_ossinsight,
    }
)

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````

## Fetching GitHub Insight Data using Function Call

This part of the notebook demonstrates the practical application of the defined functions and the Ollama Assistant Agent in fetching and interpreting GitHub Insight data.
"""
logger.info("## Fetching GitHub Insight Data using Function Call")

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
)

user_proxy.initiate_chat(oss_analyst, message="Top 10 developers with the most followers")
oss_analyst.delete_assistant()

logger.info("\n\n[DONE]", bright=True)
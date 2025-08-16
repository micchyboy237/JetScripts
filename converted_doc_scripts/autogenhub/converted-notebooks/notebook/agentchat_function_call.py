from IPython import get_ipython
from autogen.cache import Cache
from jet.logger import CustomLogger
from typing_extensions import Annotated
import autogen
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Auto Generated Agent Chat: Task Solving with Provided Tools as Functions

AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation. Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

In this notebook, we demonstrate how to use `AssistantAgent` and `UserProxyAgent` to make function calls with the new feature of Ollama models (in model version 0613). A specified prompt and function configs must be passed to `AssistantAgent` to initialize the agent. The corresponding functions must be passed to `UserProxyAgent`, which will execute any function calls made by `AssistantAgent`. Besides this requirement of matching descriptions with functions, we recommend checking the system message in the `AssistantAgent` to ensure the instructions align with the function call descriptions.

## Requirements

AutoGen requires `Python>=3.8`. To run this notebook example, please install `pyautogen`:
```bash
pip install pyautogen
```
"""
logger.info("# Auto Generated Agent Chat: Task Solving with Provided Tools as Functions")



"""
## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("## Set your API Endpoint")



config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"tags": ["tool"]},  # comment out to get all
)

"""
It first looks for environment variable "OAI_CONFIG_LIST" which needs to be a valid json string. If that variable is not found, it then looks for a json file named "OAI_CONFIG_LIST". It filters the configs by tags (you can filter by other keys as well). Only the configs with matching tags are kept in the list based on the filter condition.

The config list looks like the following:
```python
config_list = [
    {
        'model': 'gpt-4',
        'api_key': '<your Ollama API key here>',
        'tags': ['tool', 'gpt-4'],
    },
    {
        'model': 'gpt-3.5-turbo',
        'api_key': '<your Azure Ollama API key here>',
        'base_url': '<your Azure Ollama API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01
        'tags': ['tool', 'gpt-3.5-turbo'],
    },
    {
        'model': 'gpt-3.5-turbo-16k',
        'api_key': '<your Azure Ollama API key here>',
        'base_url': '<your Azure Ollama API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01
        'tags': ['tool', 'gpt-3.5-turbo-16k'],
    },
]
```

You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/microsoft/autogen/blob/main/website/docs/topics/llm_configuration.ipynb) for full code examples of the different methods.

## Making Function Calls

In this example, we demonstrate function call execution with `AssistantAgent` and `UserProxyAgent`. With the default system prompt of `AssistantAgent`, we allow the LLM assistant to perform tasks with code, and the `UserProxyAgent` would extract code blocks from the LLM response and execute them. With the new "function_call" feature, we define functions and specify the description of the function in the Ollama config for the `AssistantAgent`. Then we register the functions in `UserProxyAgent`.
"""
logger.info("## Making Function Calls")

llm_config = {
    "config_list": config_list,
    "timeout": 120,
}
chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)




@user_proxy.register_for_execution()
@chatbot.register_for_llm(name="python", description="run cell in ipython and return the execution result.")
def exec_python(cell: Annotated[str, "Valid Python cell to execute."]) -> str:
    ipython = get_ipython()
    result = ipython.run_cell(cell)
    log = str(result.result)
    if result.error_before_exec is not None:
        log += f"\n{result.error_before_exec}"
    if result.error_in_exec is not None:
        log += f"\n{result.error_in_exec}"
    return log


def exec_sh(script: Annotated[str, "Valid Python cell to execute."]) -> str:
    return user_proxy.execute_code_blocks([("sh", script)])


autogen.agentchat.register_function(
    exec_python,
    caller=chatbot,
    executor=user_proxy,
    name="sh",
    description="run a shell script and return the execution result.",
)

"""
Finally, we initialize the chat that would use the functions defined above:
"""
logger.info("Finally, we initialize the chat that would use the functions defined above:")

with Cache.disk() as cache:
    user_proxy.initiate_chat(
        chatbot,
        message="Draw two agents chatting with each other with an example dialog. Don't add plt.show().",
        cache=cache,
        max_turns=3,
    )

logger.info("\n\n[DONE]", bright=True)
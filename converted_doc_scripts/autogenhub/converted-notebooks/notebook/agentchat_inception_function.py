from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.code_utils import execute_code
from jet.logger import CustomLogger
import json
import os
import shutil
import subprocess


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Auto Generated Agent Chat: Function Inception

AutoGen offers conversable agents powered by LLM, tool or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation. Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

In this notebook, we demonstrate how to use `AssistantAgent` and `UserProxyAgent` to give them the ability to auto-extend the list of functions the model may call. Functions need to be registered to `UserProxyAgent`, which will be responsible for executing any function calls made by `AssistantAgent`. The assistant also needs to know the signature of functions that may be called. A special `define_function` function is registered, which registers a new function in `UserProxyAgent` and updates the configuration of the assistant.

In the example scenario, the user first asks to define a function that gets a URL and prints the response body. Then the user asks to print the response body, and the assistant suggests to the user to call the new function.

## Requirements

AutoGen requires `Python>=3.8`.
"""
logger.info("# Auto Generated Agent Chat: Function Inception")



config_list = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-3.5-turbo-16k-0613"],
    },
)
llm_config = {
    "functions": [
        {
            "name": "define_function",
            "description": "Define a function to add to the context of the conversation. Necessary Python packages must be declared. Once defined, the assistant may decide to use this function, respond with a normal message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the function to define.",
                    },
                    "description": {
                        "type": "string",
                        "description": "A short description of the function.",
                    },
                    "arguments": {
                        "type": "string",
                        "description": 'JSON schema of arguments encoded as a string. For example: { "url": { "type": "string", "description": "The URL", }}',
                    },
                    "packages": {
                        "type": "string",
                        "description": "A list of package names imported by the function, and that need to be installed with pip prior to invoking the function. This solves ModuleNotFoundError.",
                    },
                    "code": {
                        "type": "string",
                        "description": "The implementation in Python. Do not include the function declaration.",
                    },
                },
                "required": ["name", "description", "arguments", "packages", "code"],
            },
        },
    ],
    "config_list": config_list,
    "request_timeout": 120,
}


def define_function(name, description, arguments, packages, code):
    json_args = json.loads(arguments)
    function_config = {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": json_args},
        "required": ["url"],
    }
    llm_config["functions"] = llm_config["functions"] + [function_config]
    user_proxy.register_function(function_map={name: lambda **args: execute_func(name, packages, code, **args)})
    assistant.update_function_signature(function_config, is_remove=False)
    return f"A function has been added to the context of this conversation.\nDescription: {description}"


def execute_func(name, packages, code, **args):
    pip_install = (
        f"""logger.debug("Installing package: {packages}")\nsubprocess.run(["pip", "-qq", "install", "{packages}"])"""
        if packages
        else ""
    )
    str = f"""
{pip_install}
logger.debug("Result of {name} function execution:")
{code}
args={args}
result={name}(**args)
if result is not None: logger.debug(result)
"""
    logger.debug(f"execute_code:\n{str}")
    result = execute_code(str)[1]
    logger.debug(f"Result: {result}")
    return result


def _is_termination_msg(message):
    """Check if a message is a termination message."""
    if isinstance(message, dict):
        message = message.get("content")
        if message is None:
            return False
        return message.rstrip().endswith("TERMINATE")


assistant = AssistantAgent(
    name="chatbot",
    system_message="""You are an assistant.
        The user will ask a question.
        You may use the provided functions before providing a final answer.
        Only use the functions you were provided.
        When the answer has been provided, reply TERMINATE.""",
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config=False,
    is_termination_msg=_is_termination_msg,
    default_auto_reply="Reply TERMINATE when the initial request has been fulfilled.",
    human_input_mode="NEVER",
)

user_proxy.register_function(function_map={"define_function": define_function})


user_proxy.initiate_chat(
    assistant,
    message="Define a function that gets a URL, then prints the response body.\nReply TERMINATE when the function is defined.",
)


user_proxy.initiate_chat(
    assistant, message="Print the response body of https://echo.free.beeceptor.com/\nUse the functions you know about."
)

logger.info("\n\n[DONE]", bright=True)
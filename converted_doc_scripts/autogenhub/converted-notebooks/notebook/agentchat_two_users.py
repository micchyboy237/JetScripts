from jet.logger import CustomLogger
import autogen
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Auto Generated Agent Chat: Collaborative Task Solving with Multiple Agents and Human Users

AutoGen offers conversable agents powered by LLM, tool, or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation. Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

In this notebook, we demonstrate an application involving multiple agents and human users to work together and accomplish a task. `AssistantAgent` is an LLM-based agent that can write Python code (in a Python coding block) for a user to execute for a given task. `UserProxyAgent` is an agent which serves as a proxy for a user to execute the code written by `AssistantAgent`. We create multiple `UserProxyAgent` instances that can represent different human users.

## Requirements

AutoGen requires `Python>=3.8`. To run this notebook example, please install:
```bash
pip install autogen
```
"""
logger.info("# Auto Generated Agent Chat: Collaborative Task Solving with Multiple Agents and Human Users")



"""
## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.

It first looks for an environment variable of a specified name ("OAI_CONFIG_LIST" in this example), which needs to be a valid json string. If that variable is not found, it looks for a json file with the same name. It filters the configs by models (you can filter by other keys as well).

The json looks like the following:
```json
[
    {
        "model": "gpt-4",
        "api_key": "<your MLX API key here>"
    },
    {
        "model": "gpt-4",
        "api_key": "<your Azure MLX API key here>",
        "base_url": "<your Azure MLX API base here>",
        "api_type": "azure",
        "api_version": "2024-02-01"
    },
    {
        "model": "gpt-4-32k",
        "api_key": "<your Azure MLX API key here>",
        "base_url": "<your Azure MLX API base here>",
        "api_type": "azure",
        "api_version": "2024-02-01"
    }
]
```

You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/autogenhub/autogen/blob/main/website/docs/topics/llm_configuration.ipynb) for full code examples of the different methods.
"""
logger.info("## Set your API Endpoint")


config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

"""
## Construct Agents

We define `ask_expert` function to start a conversation between two agents and return a summary of the result. We construct an assistant agent named "assistant_for_expert" and a user proxy agent named "expert". We specify `human_input_mode` as "ALWAYS" in the user proxy agent, which will always ask for feedback from the expert user.
"""
logger.info("## Construct Agents")

def ask_expert(message):
    assistant_for_expert = autogen.AssistantAgent(
        name="assistant_for_expert",
        llm_config={
            "temperature": 0,
            "config_list": config_list,
        },
    )
    expert = autogen.UserProxyAgent(
        name="expert",
        human_input_mode="ALWAYS",
        code_execution_config={
            "work_dir": "expert",
            "use_docker": False,
        },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    )

    expert.initiate_chat(assistant_for_expert, message=message)
    expert.stop_reply_at_receive(assistant_for_expert)
    expert.send("summarize the solution and explain the answer in an easy-to-understand way", assistant_for_expert)
    return expert.last_message()["content"]

"""
We construct another assistant agent named "assistant_for_student" and a user proxy agent named "student". We specify `human_input_mode` as "TERMINATE" in the user proxy agent, which will ask for feedback when it receives a "TERMINATE" signal from the assistant agent. We set the `functions` in `AssistantAgent` and `function_map` in `UserProxyAgent` to use the created `ask_expert` function.

For simplicity, the `ask_expert` function is defined to run locally. For real applications, the function should run remotely to interact with an expert user.
"""
logger.info("We construct another assistant agent named "assistant_for_student" and a user proxy agent named "student". We specify `human_input_mode` as "TERMINATE" in the user proxy agent, which will ask for feedback when it receives a "TERMINATE" signal from the assistant agent. We set the `functions` in `AssistantAgent` and `function_map` in `UserProxyAgent` to use the created `ask_expert` function.")

assistant_for_student = autogen.AssistantAgent(
    name="assistant_for_student",
    system_message="You are a helpful assistant. Reply TERMINATE when the task is done.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
        "temperature": 0,
        "functions": [
            {
                "name": "ask_expert",
                "description": "ask expert when you can't solve the problem satisfactorily.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "question to ask expert. Ensure the question includes enough context, such as the code and the execution result. The expert does not know the conversation between you and the user unless you share the conversation with the expert.",
                        },
                    },
                    "required": ["message"],
                },
            }
        ],
    },
)

student = autogen.UserProxyAgent(
    name="student",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "student",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    function_map={"ask_expert": ask_expert},
)

"""
## Perform a task

We invoke the `initiate_chat()` method of the student proxy agent to start the conversation. When you run the cell below, you will be prompted to provide feedback after the assistant agent sends a "TERMINATE" signal at the end of the message. The conversation will finish if you don't provide any feedback (by pressing Enter directly). Before the "TERMINATE" signal, the student proxy agent will try to execute the code suggested by the assistant agent on behalf of the user.
"""
logger.info("## Perform a task")

student.initiate_chat(
    assistant_for_student,
    message="""Find $a + b + c$, given that $x+y \\neq -1$ and
\\begin{align}
	ax + by + c & = x + 7,\\
	a + bx + cy & = 2x + 6y,\\
	ay + b + cx & = 4x + y.
\\end{align}.
""",
)

"""
When the assistant needs to consult the expert, it suggests a function call to `ask_expert`. When this happens, a line like the following will be displayed:

***** Suggested function Call: ask_expert *****
"""
logger.info("When the assistant needs to consult the expert, it suggests a function call to `ask_expert`. When this happens, a line like the following will be displayed:")

logger.info("\n\n[DONE]", bright=True)
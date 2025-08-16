from IPython.display import display
from PIL import Image
from autogen.agentchat import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from jet.logger import CustomLogger
import autogen
import io
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Auto Generated Agent Chat: GPTAssistant with Code Interpreter
The latest released Assistants API by Ollama allows users to build AI assistants within their own applications. The Assistants API currently supports three types of tools: Code Interpreter, Retrieval, and Function calling. In this notebook, we demonstrate how to enable `GPTAssistantAgent` to use code interpreter. 

## Requirements

AutoGen requires `Python>=3.8`. To run this notebook example, please install:
````{=mdx}
:::info Requirements
Install `autogen`:
```bash
pip install autogen
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````

## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("# Auto Generated Agent Chat: GPTAssistant with Code Interpreter")




config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    file_location=".",
    filter_dict={
        "model": ["gpt-3.5-turbo", "gpt-35-turbo", "gpt-4", "gpt4", "gpt-4-32k", "gpt-4-turbo"],
    },
)

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````

## Perform Tasks Using Code Interpreter

We demonstrate task solving using `GPTAssistantAgent` with code interpreter. Pass `code_interpreter` in `tools` parameter to enable `GPTAssistantAgent` with code interpreter. It will write code and automatically execute it in a sandbox. The agent will receive the results from the sandbox environment and act accordingly.

### Example 1: Math Problem Solving
In this example, we demonstrate how to use code interpreter to solve math problems.
"""
logger.info("## Perform Tasks Using Code Interpreter")

gpt_assistant = GPTAssistantAgent(
    name="Coder Assistant",
    llm_config={
        "config_list": config_list,
    },
    assistant_config={
        "tools": [{"type": "code_interpreter"}],
    },
    instructions="You are an expert at solving math questions. Write code and run it to solve math problems. Reply TERMINATE when the task is solved and there is no problem.",
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    },
    human_input_mode="NEVER",
)

user_proxy.initiate_chat(
    gpt_assistant, message="If $725x + 727y = 1500$ and $729x+ 731y = 1508$, what is the value of $x - y$ ?"
)
gpt_assistant.delete_assistant()

"""
### Example 2: Plotting with Code Interpreter

Code Interpreter can outputs files, such as generating image diagrams. In this example, we demonstrate how to draw figures and download it.
"""
logger.info("### Example 2: Plotting with Code Interpreter")

gpt_assistant = GPTAssistantAgent(
    name="Coder Assistant",
    llm_config={
        "config_list": config_list,
    },
    assistant_config={
        "tools": [{"type": "code_interpreter"}],
    },
    instructions="You are an expert at writing python code to solve problems. Reply TERMINATE when the task is solved and there is no problem.",
)

user_proxy.initiate_chat(
    gpt_assistant,
    message="Draw a line chart to show the population trend in US. Show how you solved it with code.",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    clear_history=True,
    max_consecutive_auto_reply=1,
)

"""
Now we have the file id. We can download and display it.
"""
logger.info("Now we have the file id. We can download and display it.")

api_response = gpt_assistant.openai_client.files.with_raw_response.retrieve_content(
    "assistant-tvLtfOn6uAJ9kxmnxgK2OXID"
)

if api_response.status_code == 200:
    content = api_response.content
    image_data_bytes = io.BytesIO(content)
    image = Image.open(image_data_bytes)
    display(image)

gpt_assistant.delete_assistant()

logger.info("\n\n[DONE]", bright=True)
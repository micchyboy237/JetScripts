from PIL import Image
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.img_utils import _to_pil, get_image_data
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.code_utils import DEFAULT_MODEL, UNKNOWN, content_str, execute_code, extract_code, infer_lang
from jet.logger import CustomLogger
from termcolor import colored
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import autogen
import chromadb
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using Gemini in AutoGen with Other LLMs

## Installation

Install AutoGen with Gemini features:

```bash
pip install pyautogen[gemini]
```

## Dependencies of This Notebook

In this notebook, we will explore how to use Gemini in AutoGen alongside other tools. Install the necessary dependencies with the following command:

```bash
pip install pyautogen[gemini,retrievechat,lmm]
```

## Features

There's no need to handle MLX or Google's GenAI packages separately; AutoGen manages all of these for you. You can easily create different agents with various backend LLMs using the assistant agent. All models and agents are readily accessible at your fingertips. 
 

## Main Distinctions

- Currently, Gemini does not include a "system_message" field. However, you can incorporate this instruction into the first message of your interaction.
- If no API key is specified for Gemini, then authentication will happen using the default google auth mechanism for Google Cloud. Service accounts are also supported, where the JSON key file has to be provided.

Sample OAI_CONFIG_LIST 

```python
[
    {
        "model": "gpt-35-turbo",
        "api_key": "your MLX Key goes here",
    },
    {
        "model": "gpt-4-vision-preview",
        "api_key": "your MLX Key goes here",
    },
    {
        "model": "dalle",
        "api_key": "your MLX Key goes here",
    },
    {
        "model": "gemini-pro",
        "api_key": "your Google's GenAI Key goes here",
        "api_type": "google"
    },
    {
        "model": "gemini-1.5-pro-001",
        "api_type": "google"
    },
    {
        "model": "gemini-1.5-pro",
        "project_id": "your-awesome-google-cloud-project-id",
        "location": "us-west1",
        "google_application_credentials": "your-google-service-account-key.json"
    },
    {
        "model": "gemini-pro-vision",
        "api_key": "your Google's GenAI Key goes here",
        "api_type": "google"
    }
]
```
"""
logger.info("# Using Gemini in AutoGen with Other LLMs")




config_list_4v = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4-vision-preview"],
    },
)

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

config_list_gemini = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-pro-001"],
    },
)

config_list_gemini_vision = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gemini-pro-vision"],
    },
)

seed = 25  # for caching

"""
## Gemini Assistant
"""
logger.info("## Gemini Assistant")

assistant = AssistantAgent(
    "assistant", llm_config={"config_list": config_list_gemini, "seed": seed}, max_consecutive_auto_reply=3
)

user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER",
    is_termination_msg=lambda x: content_str(x.get("content")).find("TERMINATE") >= 0,
)

result = user_proxy.initiate_chat(assistant, message="Sort the array with Bubble Sort: [4, 1, 5, 2, 3]")

result

"""
## Agent Collaboration and Interactions
"""
logger.info("## Agent Collaboration and Interactions")

gpt = AssistantAgent(
    "GPT-4",
    system_message="""You should ask weird, tricky, and concise questions.
Ask the next question based on (by evolving) the previous one.""",
    llm_config={"config_list": config_list_gpt4, "seed": seed},
    max_consecutive_auto_reply=3,
)

gemini = AssistantAgent(
    "Gemini-Pro",
    system_message="""Always answer questions within one sentence. """,
    llm_config={"config_list": config_list_gemini, "seed": seed},
    max_consecutive_auto_reply=4,
)


gpt.initiate_chat(gemini, message="Do Transformers purchase auto insurance or health insurance?")

"""
Let's switch position. Now, Gemini is the question raiser. 

This time, Gemini could not follow the system instruction well or evolve questions, because the Gemini does not handle system messages similar to GPTs.
"""
logger.info("Let's switch position. Now, Gemini is the question raiser.")

gpt = AssistantAgent(
    "GPT-4",
    system_message="""Always answer questions within one sentence. """,
    llm_config={"config_list": config_list_gpt4, "seed": seed},
    max_consecutive_auto_reply=3,
)

gemini = AssistantAgent(
    "Gemini-Pro",
    system_message="""You should ask weird, tricky, and concise questions.
Ask the next question based on (by evolving) the previous one.""",
    llm_config={"config_list": config_list_gemini, "seed": seed},
    max_consecutive_auto_reply=4,
)

gemini.initiate_chat(gpt, message="Should Spider Man invest in 401K?")

"""
## Gemini Multimodal

You can create multimodal agent for Gemini the same way as the GPT-4V and LLaVA.


Note that the Gemini-pro-vision does not support chat yet. So, we only use the last message in the prompt for multi-turn chat. The behavior might be strange compared to GPT-4V and LLaVA models.

Here, we ask a question about 
![](https://github.com/microsoft/autogen/blob/main/website/static/img/chat_example.png?raw=true)
"""
logger.info("## Gemini Multimodal")

image_agent = MultimodalConversableAgent(
    "Gemini Vision", llm_config={"config_list": config_list_gemini_vision, "seed": seed}, max_consecutive_auto_reply=1
)

user_proxy = UserProxyAgent("user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=0)

user_proxy.initiate_chat(
    image_agent,
    message="""Describe what is in this image?
<img https://github.com/microsoft/autogen/blob/main/website/static/img/chat_example.png?raw=true>.""",
)

"""
## GroupChat with Gemini and GPT Agents
"""
logger.info("## GroupChat with Gemini and GPT Agents")

agent1 = AssistantAgent(
    "Gemini-agent",
    llm_config={"config_list": config_list_gemini, "seed": seed},
    max_consecutive_auto_reply=1,
    system_message="Answer questions about Google.",
    description="I am good at answering questions about Google and Research papers.",
)

agent2 = AssistantAgent(
    "GPT-agent",
    llm_config={"config_list": config_list_gpt4, "seed": seed},
    max_consecutive_auto_reply=1,
    description="I am good at writing code.",
)

user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    is_termination_msg=lambda x: content_str(x.get("content")).find("TERMINATE") >= 0
    or content_str(x.get("content")) == "",
    description="I stands for user, and can run code.",
)

groupchat = autogen.GroupChat(agents=[agent1, agent2, user_proxy], messages=[], max_round=10)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list_gemini, "seed": seed})

user_proxy.send(
    "Show me the release year of famous Google products in a markdown table.", recipient=manager, request_reply=True
)

user_proxy.send(
    "Plot the products (as y-axis) and years (as x-axis) in scatter plot and save to `graph.png`",
    recipient=manager,
    request_reply=True,
)

Image.open("coding/graph.png")

"""
## A Larger Example of Group Chat
"""
logger.info("## A Larger Example of Group Chat")

coder = AssistantAgent(
    name="Coder",
    llm_config={"config_list": config_list_gemini, "seed": seed},
    max_consecutive_auto_reply=10,
    description="I am good at writing code",
)

pm = AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config={"config_list": config_list_gemini, "seed": seed},
    max_consecutive_auto_reply=10,
    description="I am good at design products and software.",
)

user_proxy = UserProxyAgent(
    name="User_proxy",
    code_execution_config={"last_n_messages": 20, "work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER",
    is_termination_msg=lambda x: content_str(x.get("content")).find("TERMINATE") >= 0,
    description="I stands for user, and can run code.",
)

groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list_gemini, "seed": seed},
    is_termination_msg=lambda x: content_str(x.get("content")).find("TERMINATE") >= 0,
)
user_proxy.initiate_chat(
    manager,
    message="""Design and implement a multimodal product for people with vision disabilities.
The pipeline will take an image and run Gemini model to describe:
1. what objects are in the image, and
2. where these objects are located.""",
)

logger.info("\n\n[DONE]", bright=True)
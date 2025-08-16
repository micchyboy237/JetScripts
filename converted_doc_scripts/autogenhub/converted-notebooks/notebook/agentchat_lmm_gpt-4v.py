from PIL import Image
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.vision_capability import VisionCapability
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.code_utils import content_str
from jet.logger import CustomLogger
from termcolor import colored
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import autogen
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import requests
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Engaging with Multimodal Models: GPT-4V in AutoGen

In AutoGen, leveraging multimodal models can be done through two different methodologies:
1. **MultimodalAgent**: Supported by GPT-4V and other LMMs, this agent is endowed with visual cognitive abilities, allowing it to engage in interactions comparable to those of other ConversableAgents.
2. **VisionCapability**: For LLM-based agents lacking inherent visual comprehension, we introduce vision capabilities by converting images into descriptive captions.

This guide will delve into each approach, providing insights into their application and integration.

### Before everything starts, install AutoGen with the `lmm` option

Install `pyautogen`:
```bash
pip install "pyautogen[lmm]>=0.2.17"
```

For more information, please refer to the [installation guide](/docs/installation/).
"""
logger.info("# Engaging with Multimodal Models: GPT-4V in AutoGen")




"""
<a id="app-1"></a>
## Application 1: Image Chat

In this section, we present a straightforward dual-agent architecture to enable user to chat with a multimodal agent.


First, we show this image and ask a question.
![](https://th.bing.com/th/id/R.422068ce8af4e15b0634fe2540adea7a?rik=y4OcXBE%2fqutDOw&pid=ImgRaw&r=0)

Within the user proxy agent, we can decide to activate the human input mode or not (for here, we use human_input_mode="NEVER" for conciseness). This allows you to interact with LMM in a multi-round dialogue, enabling you to provide feedback as the conversation unfolds.
"""
logger.info("## Application 1: Image Chat")

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

gpt4_llm_config = {"config_list": config_list_gpt4, "cache_seed": 42}

"""
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
"""
logger.info("Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).")

image_agent = MultimodalConversableAgent(
    name="image-explainer",
    max_consecutive_auto_reply=10,
    llm_config={"config_list": config_list_4v, "temperature": 0.5, "max_tokens": 300},
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    human_input_mode="NEVER",  # Try between ALWAYS or NEVER
    max_consecutive_auto_reply=0,
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

user_proxy.initiate_chat(
    image_agent,
    message="""What's the breed of this dog?
<img https://th.bing.com/th/id/R.422068ce8af4e15b0634fe2540adea7a?rik=y4OcXBE%2fqutDOw&pid=ImgRaw&r=0>.""",
)

"""
Now, input another image, and ask a followup question.

![](https://th.bing.com/th/id/OIP.29Mi2kJmcHHyQVGe_0NG7QHaEo?pid=ImgDet&rs=1)
"""
logger.info("Now, input another image, and ask a followup question.")

user_proxy.send(
    message="""What is this breed?
<img https://th.bing.com/th/id/OIP.29Mi2kJmcHHyQVGe_0NG7QHaEo?pid=ImgDet&rs=1>

Among the breeds, which one barks less?""",
    recipient=image_agent,
)

"""
<a id="app-2"></a>
## Application 2: Figure Creator

Here, we define a `FigureCreator` agent, which contains three child agents: commander, coder, and critics.

- Commander: interacts with users, runs code, and coordinates the flow between the coder and critics.
- Coder: writes code for visualization.
- Critics: LMM-based agent that provides comments and feedback on the generated image.
"""
logger.info("## Application 2: Figure Creator")

working_dir = "tmp/"

class FigureCreator(ConversableAgent):
    def __init__(self, n_iters=2, **kwargs):
        """
        Initializes a FigureCreator instance.

        This agent facilitates the creation of visualizations through a collaborative effort among its child agents: commander, coder, and critics.

        Parameters:
            - n_iters (int, optional): The number of "improvement" iterations to run. Defaults to 2.
            - **kwargs: keyword arguments for the parent AssistantAgent.
        """
        super().__init__(**kwargs)
        self.register_reply([Agent, None], reply_func=FigureCreator._reply_user, position=0)
        self._n_iters = n_iters

    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)  # noqa: F821
            raise AssertionError(error_msg)
        if messages is None:
            messages = self._oai_messages[sender]

        user_question = messages[-1]["content"]

        commander = AssistantAgent(
            name="Commander",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            system_message="Help me run the code, and tell other agents it is in the <img result.jpg> file location.",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={"last_n_messages": 3, "work_dir": working_dir, "use_docker": False},
            llm_config=self.llm_config,
        )

        critics = MultimodalConversableAgent(
            name="Critics",
            system_message="""Criticize the input figure. How to replot the figure so it will be better? Find bugs and issues for the figure.
            Pay attention to the color, format, and presentation. Keep in mind of the reader-friendliness.
            If you think the figures is good enough, then simply say NO_ISSUES""",
            llm_config={"config_list": config_list_4v, "max_tokens": 300},
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

        coder = AssistantAgent(
            name="Coder",
            llm_config=self.llm_config,
        )

        coder.update_system_message(
            coder.system_message
            + "ALWAYS save the figure in `result.jpg` file. Tell other agents it is in the <img result.jpg> file location."
        )

        commander.initiate_chat(coder, message=user_question)
        img = Image.open(os.path.join(working_dir, "result.jpg"))
        plt.imshow(img)
        plt.axis("off")  # Hide the axes
        plt.show()

        for i in range(self._n_iters):
            commander.send(
                message=f"Improve <img {os.path.join(working_dir, 'result.jpg')}>",
                recipient=critics,
                request_reply=True,
            )

            feedback = commander._oai_messages[critics][-1]["content"]
            if feedback.find("NO_ISSUES") >= 0:
                break
            commander.send(
                message="Here is the feedback to your figure. Please improve! Save the result to `result.jpg`\n"
                + feedback,
                recipient=coder,
                request_reply=True,
            )
            img = Image.open(os.path.join(working_dir, "result.jpg"))
            plt.imshow(img)
            plt.axis("off")  # Hide the axes
            plt.show()

        return True, os.path.join(working_dir, "result.jpg")

creator = FigureCreator(name="Figure Creator~", llm_config=gpt4_llm_config)

user_proxy = autogen.UserProxyAgent(
    name="User", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config={"use_docker": False}
)

user_proxy.initiate_chat(
    creator,
    message="""
Plot a figure by using the data from:
https://raw.githubusercontent.com/vega/vega/main/docs/data/seattle-weather.csv

I want to show both temperature high and low.
""",
)

"""
## Vision Capability: Group Chat Example with Multimodal Agent

We recommend using VisionCapability for group chat managers so that it can organize and understand images better.
"""
logger.info("## Vision Capability: Group Chat Example with Multimodal Agent")

agent1 = MultimodalConversableAgent(
    name="image-explainer-1",
    max_consecutive_auto_reply=10,
    llm_config={"config_list": config_list_4v, "temperature": 0.5, "max_tokens": 300},
    system_message="Your image description is poetic and engaging.",
)
agent2 = MultimodalConversableAgent(
    name="image-explainer-2",
    max_consecutive_auto_reply=10,
    llm_config={"config_list": config_list_4v, "temperature": 0.5, "max_tokens": 300},
    system_message="Your image description is factual and to the point.",
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="Desribe image for me.",
    human_input_mode="TERMINATE",  # Try between ALWAYS, NEVER, and TERMINATE
    max_consecutive_auto_reply=10,
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

groupchat = autogen.GroupChat(agents=[agent1, agent2, user_proxy], messages=[], max_round=5)

vision_capability = VisionCapability(lmm_config={"config_list": config_list_4v, "temperature": 0.5, "max_tokens": 300})
group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_llm_config)
vision_capability.add_to_agent(group_chat_manager)

rst = user_proxy.initiate_chat(
    group_chat_manager,
    message="""Write a poet for my image:
                        <img https://th.bing.com/th/id/R.422068ce8af4e15b0634fe2540adea7a?rik=y4OcXBE%2fqutDOw&pid=ImgRaw&r=0>.""",
)

"""
## Behavior with and without VisionCapability for Agents


Here, we show the behavior of an agent with and without VisionCapability. We use the same image and question as in the previous example.
"""
logger.info("## Behavior with and without VisionCapability for Agents")

agent_no_vision = AssistantAgent(name="Regular LLM Agent", llm_config=gpt4_llm_config)

agent_with_vision = AssistantAgent(name="Regular LLM Agent with Vision Capability", llm_config=gpt4_llm_config)
vision_capability = VisionCapability(lmm_config={"config_list": config_list_4v, "temperature": 0.5, "max_tokens": 300})
vision_capability.add_to_agent(agent_with_vision)


user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config={"use_docker": False},
)

message = """Write a poet for my image:
                        <img https://th.bing.com/th/id/R.422068ce8af4e15b0634fe2540adea7a?rik=y4OcXBE%2fqutDOw&pid=ImgRaw&r=0>."""

user.send(message=message, recipient=agent_no_vision, request_reply=True)

user.send(message=message, recipient=agent_with_vision, request_reply=True)

"""
## Custom Caption Function for Vision Capability

In many use cases, we can use a custom function within the Vision Capability to transcribe an image into a caption.

For instance, we can use rule-based algorithm or other models to detect the color, box, and other components inside the image.

The custom model should take a path to the image and return a string caption.

In the example below, the Vision Capability will call LMM to get caption and also call the custom function to get more information.
"""
logger.info("## Custom Caption Function for Vision Capability")

def my_description(image_url: str, image_data: Image = None, lmm_client: object = None) -> str:
    """
    This function takes an image URL and returns the description.

    Parameters:
        - image_url (str): The URL of the image.
        - image_data (PIL.Image): The image data.
        - lmm_client (object): The LLM client object.

    Returns:
        - str: A description of the color of the image.
    """
    logger.debug("image_url", image_url)
    logger.debug("image_data", image_data)
    logger.debug("lmm_client", lmm_client)

    img_uri = pil_to_data_uri(image_data)  # cast data into URI (str) format for API call
    lmm_out = lmm_client.create(
        context=None,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in 10 words."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_uri,
                        },
                    },
                ],
            }
        ],
    )
    description = lmm_out.choices[0].message.content
    description = content_str(description)

    pixels = np.array(image_data)

    avg_color_per_row = np.mean(pixels, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)
    avg_color = avg_color.astype(int)  # Convert to integer for color values

    caption = f"""The image is from {image_url}
    It is about: {description}
    The average color of the image is RGB:
        ({avg_color[0]}, {avg_color[1]}, {avg_color[2]})"""

    logger.debug(caption)  # For illustration purpose

    return caption

agent_with_vision_and_func = AssistantAgent(
    name="Regular LLM Agent with Custom Func and LMM", llm_config=gpt4_llm_config
)

vision_capability_with_func = VisionCapability(
    lmm_config={"config_list": config_list_4v, "temperature": 0.5, "max_tokens": 300},
    custom_caption_func=my_description,
)
vision_capability_with_func.add_to_agent(agent_with_vision_and_func)

user.send(message=message, recipient=agent_with_vision_and_func, request_reply=True)

logger.info("\n\n[DONE]", bright=True)
from PIL import Image
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.llava_agent import LLaVAAgent, llava_call
from jet.logger import CustomLogger
from termcolor import colored
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import autogen
import json
import matplotlib.pyplot as plt
import os
import random
import replicate
import requests
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Agent Chat with Multimodal Models: LLaVA

This notebook uses **LLaVA** as an example for the multimodal feature. More information about LLaVA can be found in their [GitHub page](https://github.com/haotian-liu/LLaVA)


This notebook contains the following information and examples:

1. Setup LLaVA Model
    - Option 1: Use [API calls from `Replicate`](#replicate)
    - Option 2: Setup [LLaVA locally (requires GPU)](#local)
2. Application 1: [Image Chat](#app-1)
3. Application 2: [Figure Creator](#app-2)

### Before everything starts, install AutoGen with the `lmm` option
```bash
pip install "autogen[lmm]>=0.3.0"
```
"""
logger.info("# Agent Chat with Multimodal Models: LLaVA")




LLAVA_MODE = "remote"  # Either "local" or "remote"
assert LLAVA_MODE in ["local", "remote"]

"""
<a id="replicate"></a>
## (Option 1, preferred) Use API Calls from Replicate [Remote]
We can also use [Replicate](https://replicate.com/yorickvp/llava-13b/api) to use LLaVA directly, which will host the model for you.

1. Run `pip install replicate` to install the package
2. You need to get an API key from Replicate from your [account setting page](https://replicate.com/account/api-tokens)
3. Next, copy your API token and authenticate by setting it as an environment variable:
    `export REPLICATE_API_TOKEN=<paste-your-token-here>` 
4. You need to enter your credit card information for Replicate 🥲
"""
logger.info("## (Option 1, preferred) Use API Calls from Replicate [Remote]")



if LLAVA_MODE == "remote":

    llava_config_list = [
        {
            "model": "whatever, will be ignored for remote",  # The model name doesn't matter here right now.
            "api_key": "None",  # Note that you have to setup the API key with os.environ["REPLICATE_API_TOKEN"]
            "base_url": "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
        }
    ]

"""
<a id="local"></a>
## [Option 2] Setup LLaVA Locally


## Install the LLaVA library

Please follow the LLaVA GitHub [page](https://github.com/haotian-liu/LLaVA/) to install LLaVA.


#### Download the package
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

#### Install the inference package
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```



Some helpful packages and dependencies:
```bash
conda install -c nvidia cuda-toolkit
```


### Launch

In one terminal, start the controller first:
```bash
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```


Then, in another terminal, start the worker, which will load the model to the GPU:
```bash
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-13b
``
"""
logger.info("## [Option 2] Setup LLaVA Locally")

if LLAVA_MODE == "local":
    llava_config_list = [
        {
            "model": "llava-v1.5-13b",
            "api_key": "None",
            "base_url": "http://0.0.0.0:10000",
        }
    ]

"""
# Multimodal Functions

We cal test the `llava_call` function with the following AutoGen image.
![](https://raw.githubusercontent.com/microsoft/autogen/main/website/static/img/autogen_agentchat.png)
"""
logger.info("# Multimodal Functions")

rst = llava_call(
    "Describe this AutoGen framework <img https://raw.githubusercontent.com/microsoft/autogen/main/website/static/img/autogen_agentchat.png> with bullet points.",
    llm_config={"config_list": llava_config_list, "temperature": 0},
)

logger.debug(rst)

"""
<a id="app-1"></a>
## Application 1: Image Chat

In this section, we present a straightforward dual-agent architecture to enable user to chat with a multimodal agent.


First, we show this image and ask a question.
![](https://th.bing.com/th/id/R.422068ce8af4e15b0634fe2540adea7a?rik=y4OcXBE%2fqutDOw&pid=ImgRaw&r=0)

Within the user proxy agent, we can decide to activate the human input mode or not (for here, we use human_input_mode="NEVER" for conciseness). This allows you to interact with LLaVA in a multi-round dialogue, enabling you to provide feedback as the conversation unfolds.
"""
logger.info("## Application 1: Image Chat")

image_agent = LLaVAAgent(
    name="image-explainer",
    max_consecutive_auto_reply=10,
    llm_config={"config_list": llava_config_list, "temperature": 0.5, "max_new_tokens": 1000},
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "groupchat",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    human_input_mode="NEVER",  # Try between ALWAYS or NEVER
    max_consecutive_auto_reply=0,
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
- Critics: LLaVA-based agent that provides comments and feedback on the generated image.
"""
logger.info("## Application 2: Figure Creator")

class FigureCreator(AssistantAgent):
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
            code_execution_config={
                "last_n_messages": 3,
                "work_dir": ".",
                "use_docker": False,
            },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
            llm_config=self.llm_config,
        )

        critics = LLaVAAgent(
            name="Critics",
            system_message="""Criticize the input figure. How to replot the figure so it will be better? Find bugs and issues for the figure.
            Pay attention to the color, format, and presentation. Keep in mind of the reader-friendliness.
            If you think the figures is good enough, then simply say NO_ISSUES""",
            llm_config={"config_list": llava_config_list},
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
        img = Image.open("result.jpg")
        plt.imshow(img)
        plt.axis("off")  # Hide the axes
        plt.show()

        for i in range(self._n_iters):
            commander.send(message="Improve <img result.jpg>", recipient=critics, request_reply=True)

            feedback = commander._oai_messages[critics][-1]["content"]
            if feedback.find("NO_ISSUES") >= 0:
                break
            commander.send(
                message="Here is the feedback to your figure. Please improve! Save the result to `result.jpg`\n"
                + feedback,
                recipient=coder,
                request_reply=True,
            )
            img = Image.open("result.jpg")
            plt.imshow(img)
            plt.axis("off")  # Hide the axes
            plt.show()

        return True, "result.jpg"

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

gpt4_llm_config = {"config_list": config_list_gpt4, "cache_seed": 42}




creator = FigureCreator(name="Figure Creator~", llm_config=gpt4_llm_config)

user_proxy = autogen.UserProxyAgent(
    name="User", human_input_mode="NEVER", max_consecutive_auto_reply=0, code_execution_config={"use_docker": False}
)  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.

user_proxy.initiate_chat(
    creator,
    message="""
Plot a figure by using the data from:
https://raw.githubusercontent.com/vega/vega/main/docs/data/seattle-weather.csv

I want to show both temperature high and low.
""",
)

if os.path.exists("result.jpg"):
    os.remove("result.jpg")  # clean up

logger.info("\n\n[DONE]", bright=True)
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.agentchat.contrib.capabilities.vision_capability import VisionCapability
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.code_utils import content_str
from jet.logger import CustomLogger
from typing import Annotated, Literal
from typing import Literal
from typing_extensions import Annotated
import autogen
import json
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Amazon Bedrock

AutoGen allows you to use Amazon's generative AI Bedrock service to run inference with a number of open-weight models and as well as their own models.

Amazon Bedrock supports models from providers such as Meta, Anthropic, Cohere, and Mistral.

In this notebook, we demonstrate how to use Anthropic's Sonnet model for AgentChat in AutoGen.

## Model features / support

Amazon Bedrock supports a wide range of models, not only for text generation but also for image classification and generation. Not all features are supported by AutoGen or by the Converse API used. Please see [Amazon's documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-supported-models-features) on the features supported by the Converse API.

At this point in time AutoGen supports text generation and image classification (passing images to the LLM).

It does not, yet, support image generation ([contribute](https://microsoft.github.io/autogen/docs/contributor-guide/contributing/)).

## Requirements
To use Amazon Bedrock with AutoGen, first you need to install the `pyautogen[bedrock]` package.

## Pricing

When we combine the number of models supported and costs being on a per-region basis, it's not feasible to maintain the costs for each model+region combination within the AutoGen implementation. Therefore, it's recommended that you add the following to your config with cost per 1,000 input and output tokens, respectively:
```
{
    ...
    "price": [0.003, 0.015]
    ...
}
```

Amazon Bedrock pricing is available [here](https://aws.amazon.com/bedrock/pricing/).
"""
logger.info("# Amazon Bedrock")

# !pip install pyautogen["bedrock"]

"""
## Set the config for Amazon Bedrock

Amazon's Bedrock does not use the `api_key` as per other cloud inference providers for authentication, instead it uses a number of access, token, and profile values. These fields will need to be added to your client configuration. Please check the Amazon Bedrock documentation to determine which ones you will need to add.

The available parameters are:

- aws_region (mandatory)
- aws_access_key (or environment variable: AWS_ACCESS_KEY)
- aws_secret_key (or environment variable: AWS_SECRET_KEY)
- aws_session_token (or environment variable: AWS_SESSION_TOKEN)
- aws_profile_name

Beyond the authentication credentials, the only mandatory parameters are `api_type` and `model`.

The following parameters are common across all models used:

- temperature
- topP
- maxTokens

You can also include parameters specific to the model you are using (see the model detail within Amazon's documentation for more information), the four supported additional parameters are:

- top_p
- top_k
- k
- seed

An additional parameter can be added that denotes whether the model supports a system prompt (which is where the system messages are not included in the message list, but in a separate parameter). This defaults to `True`, so set it to `False` if your model (for example Mistral's Instruct models) [doesn't support this feature](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-supported-models-features):

- supports_system_prompts

It is important to add the `api_type` field and set it to a string that corresponds to the client type used: `bedrock`.

Example:
```
[
    {
        "api_type": "bedrock",
        "model": "amazon.titan-text-premier-v1:0",
        "aws_region": "us-east-1"
        "aws_access_key": "",
        "aws_secret_key": "",
        "aws_session_token": "",
        "aws_profile_name": "",
    },
    {
        "api_type": "bedrock",
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "aws_region": "us-east-1"
        "aws_access_key": "",
        "aws_secret_key": "",
        "aws_session_token": "",
        "aws_profile_name": "",
        "temperature": 0.5,
        "topP": 0.2,
        "maxTokens": 250,
    },
    {
        "api_type": "bedrock",
        "model": "mistral.mixtral-8x7b-instruct-v0:1",
        "aws_region": "us-east-1"
        "aws_access_key": "",
        "aws_secret_key": "",
        "supports_system_prompts": False, # Mistral Instruct models don't support a separate system prompt
        "price": [0.00045, 0.0007] # Specific pricing for this model/region
    }
]
```

## Two-agent Coding Example

### Configuration

Start with our configuration - we'll use Anthropic's Sonnet model and put in recent pricing. Additionally, we'll reduce the temperature to 0.1 so its responses are less varied.
"""
logger.info("## Set the config for Amazon Bedrock")



config_list_bedrock = [
    {
        "api_type": "bedrock",
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "aws_region": "us-east-1",
        "aws_access_key": "[FILL THIS IN]",
        "aws_secret_key": "[FILL THIS IN]",
        "price": [0.003, 0.015],
        "temperature": 0.1,
        "cache_seed": None,  # turn off caching
    }
]

"""
### Construct Agents

Construct a simple conversation between a User proxy and an ConversableAgent, which uses the Sonnet model.
"""
logger.info("### Construct Agents")

assistant = autogen.AssistantAgent(
    "assistant",
    llm_config={
        "config_list": config_list_bedrock,
    },
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", ""),
    max_consecutive_auto_reply=1,
)

"""
### Initiate Chat
"""
logger.info("### Initiate Chat")

user_proxy.initiate_chat(
    assistant,
    message="Write a python program to print the first 10 numbers of the Fibonacci sequence. Just output the python code, no additional information.",
)

"""
## Tool Call Example

In this example, instead of writing code, we will show how we can perform multiple tool calling with Meta's Llama 3.1 70B model, where it recommends calling more than one tool at a time.

We'll use a simple travel agent assistant program where we have a couple of tools for weather and currency conversion.

### Agents
"""
logger.info("## Tool Call Example")



config_list_bedrock = [
    {
        "api_type": "bedrock",
        "model": "meta.llama3-1-70b-instruct-v1:0",
        "aws_region": "us-west-2",
        "aws_access_key": "[FILL THIS IN]",
        "aws_secret_key": "[FILL THIS IN]",
        "price": [0.00265, 0.0035],
        "cache_seed": None,  # turn off caching
    }
]

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="""For currency exchange and weather forecasting tasks,
        only use the functions you have been provided with.
        Output only the word 'TERMINATE' when an answer has been provided.
        Use both tools together if you can.""",
    llm_config={
        "config_list": config_list_bedrock,
    },
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", ""),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
)

"""
Create the two functions, annotating them so that those descriptions can be passed through to the LLM.

With Meta's Llama 3.1 models, they are more likely to pass a numeric parameter as a string, e.g. "123.45" instead of 123.45, so we'll convert numeric parameters from strings to floats if necessary.

We associate them with the agents using `register_for_execution` for the user_proxy so it can execute the function and `register_for_llm` for the chatbot (powered by the LLM) so it can pass the function definitions to the LLM.
"""
logger.info("Create the two functions, annotating them so that those descriptions can be passed through to the LLM.")

CurrencySymbol = Literal["USD", "EUR"]



def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
    if base_currency == quote_currency:
        return 1.0
    elif base_currency == "USD" and quote_currency == "EUR":
        return 1 / 1.1
    elif base_currency == "EUR" and quote_currency == "USD":
        return 1.1
    else:
        raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")




@user_proxy.register_for_execution()
@chatbot.register_for_llm(description="Currency exchange calculator.")
def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency, float values (no strings), e.g. 987.82"],
    base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
    quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
) -> str:
    if isinstance(base_amount, str):
        base_amount = float(base_amount)

    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{format(quote_amount, '.2f')} {quote_currency}"




def get_current_weather(location, unit="fahrenheit"):
    """Get the weather for some location"""
    if "chicago" in location.lower():
        return json.dumps({"location": "Chicago", "temperature": "13", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "55", "unit": unit})
    elif "new york" in location.lower():
        return json.dumps({"location": "New York", "temperature": "11", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})




@user_proxy.register_for_execution()
@chatbot.register_for_llm(description="Weather forecast for US cities.")
def weather_forecast(
    location: Annotated[str, "City name"],
) -> str:
    weather_details = get_current_weather(location=location)
    weather = json.loads(weather_details)
    return f"{weather['location']} will be {weather['temperature']} degrees {weather['unit']}"

"""
We pass through our customer's message and run the chat.

Finally, we ask the LLM to summarise the chat and print that out.
"""
logger.info("We pass through our customer's message and run the chat.")

res = user_proxy.initiate_chat(
    chatbot,
    message="What's the weather in New York and can you tell me how much is 123.45 EUR in USD so I can spend it on my holiday?",
    summary_method="reflection_with_llm",
)

logger.debug(res.summary["content"])

"""
## Group Chat Example with Anthropic's Claude 3 Sonnet, Mistral's Large 2, and Meta's Llama 3.1 70B

The flexibility of using LLMs from the industry's leading providers, particularly larger models, with Amazon Bedrock allows you to use multiple of them in a single workflow.

Here we have a conversation that has two models (Anthropic's Claude 3 Sonnet and Mistral's Large 2) debate each other with another as the judge (Meta's Llama 3.1 70B). Additionally, a tool call is made to pull through some mock news that they will debate on.
"""
logger.info("## Group Chat Example with Anthropic's Claude 3 Sonnet, Mistral's Large 2, and Meta's Llama 3.1 70B")



config_list_sonnet = [
    {
        "api_type": "bedrock",
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "aws_region": "us-east-1",
        "aws_access_key": "[FILL THIS IN]",
        "aws_secret_key": "[FILL THIS IN]",
        "price": [0.003, 0.015],
        "temperature": 0.1,
        "cache_seed": None,  # turn off caching
    }
]

config_list_mistral = [
    {
        "api_type": "bedrock",
        "model": "mistral.mistral-large-2407-v1:0",
        "aws_region": "us-west-2",
        "aws_access_key": "[FILL THIS IN]",
        "aws_secret_key": "[FILL THIS IN]",
        "price": [0.003, 0.009],
        "temperature": 0.1,
        "cache_seed": None,  # turn off caching
    }
]

config_list_llama31_70b = [
    {
        "api_type": "bedrock",
        "model": "meta.llama3-1-70b-instruct-v1:0",
        "aws_region": "us-west-2",
        "aws_access_key": "[FILL THIS IN]",
        "aws_secret_key": "[FILL THIS IN]",
        "price": [0.00265, 0.0035],
        "temperature": 0.1,
        "cache_seed": None,  # turn off caching
    }
]

alice = AssistantAgent(
    "sonnet_agent",
    system_message="You are from Anthropic, an AI company that created the Sonnet large language model. You make arguments to support your company's position. You analyse given text. You are not a programmer and don't use Python. Pass to mistral_agent when you have finished. Start your response with 'I am sonnet_agent'.",
    llm_config={
        "config_list": config_list_sonnet,
    },
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

bob = autogen.AssistantAgent(
    "mistral_agent",
    system_message="You are from Mistral, an AI company that created the Large v2 large language model. You make arguments to support your company's position. You analyse given text. You are not a programmer and don't use Python. Pass to the judge if you have finished. Start your response with 'I am mistral_agent'.",
    llm_config={
        "config_list": config_list_mistral,
    },
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

charlie = AssistantAgent(
    "research_assistant",
    system_message="You are a helpful assistant to research the latest news and headlines. You have access to call functions to get the latest news articles for research through 'code_interpreter'.",
    llm_config={
        "config_list": config_list_llama31_70b,
    },
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

dan = AssistantAgent(
    "judge",
    system_message="You are a judge. You will evaluate the arguments and make a decision on which one is more convincing. End your decision with the word 'TERMINATE' to conclude the debate.",
    llm_config={
        "config_list": config_list_llama31_70b,
    },
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

code_interpreter = UserProxyAgent(
    "code_interpreter",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    default_auto_reply="",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)


@code_interpreter.register_for_execution()  # Decorator factory for registering a function to be executed by an agent
@charlie.register_for_llm(
    name="get_headlines", description="Get the headline of a particular day."
)  # Decorator factory for registering a function to be used by an agent
def get_headlines(headline_date: Annotated[str, "Date in MMDDYY format, e.g., 06192024"]) -> str:
    mock_news = {
        "06202024": """Epic Duel of the Titans: Anthropic and Mistral Usher in a New Era of Text Generation Excellence.
        In a groundbreaking revelation that has sent shockwaves through the AI industry, Anthropic has unveiled
        their state-of-the-art text generation model, Sonnet, hailed as a monumental leap in artificial intelligence.
        Almost simultaneously, Mistral countered with their equally formidable creation, Large 2, showcasing
        unparalleled prowess in generating coherent and contextually rich text. This scintillating rivalry
        between two AI behemoths promises to revolutionize the landscape of machine learning, heralding an
        era of unprecedented creativity and sophistication in text generation that will reshape industries,
        ignite innovation, and captivate minds worldwide.""",
        "06192024": "MLX founder Sutskever sets up new AI company devoted to safe superintelligence.",
    }
    return mock_news.get(headline_date, "No news available for today.")


user_proxy = UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    default_auto_reply="",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

groupchat = GroupChat(
    agents=[alice, bob, charlie, dan, code_interpreter],
    messages=[],
    allow_repeat_speaker=False,
    max_round=10,
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={
        "config_list": config_list_llama31_70b,
    },
)

task = "Analyze the potential of Anthropic and Mistral to revolutionize the field of AI based on today's headlines. Today is 06202024. Start by selecting 'research_assistant' to get relevant news articles and then ask sonnet_agent and mistral_agent to respond before the judge evaluates the conversation."

user_proxy.initiate_chat(manager, message=task)

"""
And there we have it, a number of different LLMs all collaborating together on a single cloud platform.

## Image classification with Anthropic's Claude 3 Sonnet

AutoGen's Amazon Bedrock client class supports inputting images for the LLM to respond to.

In this simple example, we'll use an image on the Internet and send it to Anthropic's Claude 3 Sonnet model to describe.

Here's the image we'll use:

![I -heart- AutoGen](https://microsoft.github.io/autogen/assets/images/love-ec54b2666729d3e9d93f91773d1a77cf.png "width=400 height=400")
"""
logger.info("## Image classification with Anthropic's Claude 3 Sonnet")

config_list_sonnet = {
    "config_list": [
        {
            "api_type": "bedrock",
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "aws_region": "us-east-1",
            "aws_access_key": "[FILL THIS IN]",
            "aws_secret_key": "[FILL THIS IN]",
            "cache_seed": None,
        }
    ]
}

"""
We'll use a Multimodal agent to handle the image
"""
logger.info("We'll use a Multimodal agent to handle the image")


image_agent = MultimodalConversableAgent(
    name="image-explainer",
    max_consecutive_auto_reply=10,
    llm_config=config_list_sonnet,
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

"""
We start the chat and use the `img` tag in the message. The image will be downloaded and converted to bytes, then sent to the LLM.
"""
logger.info("We start the chat and use the `img` tag in the message. The image will be downloaded and converted to bytes, then sent to the LLM.")

result = user_proxy.initiate_chat(
    image_agent,
    message="""What's happening in this image?
<img https://microsoft.github.io/autogen/assets/images/love-ec54b2666729d3e9d93f91773d1a77cf.png>.""",
)

logger.info("\n\n[DONE]", bright=True)
from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from typing import Dict, List, Tuple
import autogen
import copy
import pprint
import re

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Preprocessing Chat History with `TransformMessages`

## Introduction
This notebook illustrates how to use `TransformMessages` give any `ConversableAgent` the ability to handle long contexts, sensitive data, and more.

````{=mdx}
:::info Requirements
Install `autogen`:
```bash
pip install autogen
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````
"""
logger.info("# Preprocessing Chat History with `TransformMessages`")



config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
)
llm_config = {"config_list": config_list}

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````
"""
logger.info("Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).")

assistant = autogen.AssistantAgent(
    "assistant",
    llm_config=llm_config,
)
user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    max_consecutive_auto_reply=10,
)

"""
## Handling Long Contexts

Imagine a scenario where the LLM generates an extensive amount of text, surpassing the token limit imposed by your API provider. To address this issue, you can leverage `TransformMessages` along with its constituent transformations, `MessageHistoryLimiter` and `MessageTokenLimiter`.

- `MessageHistoryLimiter`: You can restrict the total number of messages considered as context history. This transform is particularly useful when you want to limit the conversational context to a specific number of recent messages, ensuring efficient processing and response generation.
- `MessageTokenLimiter`: Enables you to cap the total number of tokens, either on a per-message basis or across the entire context history (or both). This transformation is invaluable when you need to adhere to strict token limits imposed by your API provider, preventing unnecessary costs or errors caused by exceeding the allowed token count. Additionally, a `min_tokens` threshold can be applied, ensuring that the transformation is only applied when the number of tokens is not less than the specified threshold.
"""
logger.info("## Handling Long Contexts")

max_msg_transfrom = transforms.MessageHistoryLimiter(max_messages=3)

token_limit_transform = transforms.MessageTokenLimiter(max_tokens_per_message=3, min_tokens=10)

"""
## Example 1: Limiting number of messages
Let's take a look at how these transformations will effect the messages. Below we see that by applying the `MessageHistoryLimiter`, we can see that we limited the context history to the 3 most recent messages.
"""
logger.info("## Example 1: Limiting number of messages")

messages = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": [{"type": "text", "text": "there"}]},
    {"role": "user", "content": "how"},
    {"role": "assistant", "content": [{"type": "text", "text": "are you doing?"}]},
    {"role": "user", "content": "very very very very very very long string"},
]

processed_messages = max_msg_transfrom.apply_transform(copy.deepcopy(messages))
pprint.plogger.debug(processed_messages)

"""
## Example 2: Limiting number of tokens

Now let's test limiting the number of tokens in messages. We can see that we can limit the number of tokens to 3, which is equivalent to 3 words in this instance.
"""
logger.info("## Example 2: Limiting number of tokens")

processed_messages = token_limit_transform.apply_transform(copy.deepcopy(messages))

pprint.plogger.debug(processed_messages)

"""
Also, the `min_tokens` threshold is set to 10, indicating that the transformation will not be applied if the total number of tokens in the messages is less than that. This is especially beneficial when the transformation should only occur after a certain number of tokens has been reached, such as in the context window of the model. An example is provided below.
"""
logger.info("Also, the `min_tokens` threshold is set to 10, indicating that the transformation will not be applied if the total number of tokens in the messages is less than that. This is especially beneficial when the transformation should only occur after a certain number of tokens has been reached, such as in the context window of the model. An example is provided below.")

short_messages = [
    {"role": "user", "content": "hello there, how are you?"},
    {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
]

processed_short_messages = token_limit_transform.apply_transform(copy.deepcopy(short_messages))

pprint.plogger.debug(processed_short_messages)

"""
## Example 3: Combining transformations

Let's test these transforms with agents (the upcoming test is replicated from the agentchat_capability_long_context_handling notebook). We will see that the agent without the capability to handle long context will result in an error, while the agent with that capability will have no issues.
"""
logger.info("## Example 3: Combining transformations")

assistant_base = autogen.AssistantAgent(
    "assistant",
    llm_config=llm_config,
)

assistant_with_context_handling = autogen.AssistantAgent(
    "assistant",
    llm_config=llm_config,
)
context_handling = transform_messages.TransformMessages(
    transforms=[
        transforms.MessageHistoryLimiter(max_messages=10),
        transforms.MessageTokenLimiter(max_tokens=1000, max_tokens_per_message=50, min_tokens=500),
    ]
)

context_handling.add_to_agent(assistant_with_context_handling)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    max_consecutive_auto_reply=2,
)

for i in range(1000):
    assitant_msg = {"role": "assistant", "content": "test " * 1000}
    user_msg = {"role": "user", "content": ""}

    assistant_base.send(assitant_msg, user_proxy, request_reply=False, silent=True)
    assistant_with_context_handling.send(assitant_msg, user_proxy, request_reply=False, silent=True)
    user_proxy.send(user_msg, assistant_base, request_reply=False, silent=True)
    user_proxy.send(user_msg, assistant_with_context_handling, request_reply=False, silent=True)

try:
    user_proxy.initiate_chat(assistant_base, message="plot and save a graph of x^2 from -10 to 10", clear_history=False)
except Exception as e:
    logger.debug("Encountered an error with the base assistant")
    logger.debug(e)
    logger.debug("\n\n")

try:
    user_proxy.initiate_chat(
        assistant_with_context_handling, message="plot and save a graph of x^2 from -10 to 10", clear_history=False
    )
except Exception as e:
    logger.debug(e)

"""
## Handling Sensitive Data

You can use the `MessageTransform` protocol to create custom message transformations that redact sensitive data from the chat history. This is particularly useful when you want to ensure that sensitive information, such as API keys, passwords, or personal data, is not exposed in the chat history or logs.

Now, we will create a custom message transform to detect any Ollama API key and redact it.
"""
logger.info("## Handling Sensitive Data")

class MessageRedact:
    def __init__(self):
        self._openai_key_pattern = r"sk-([a-zA-Z0-9]{48})"
        self._replacement_string = "REDACTED"

    def apply_transform(self, messages: List[Dict]) -> List[Dict]:
        temp_messages = copy.deepcopy(messages)

        for message in temp_messages:
            if isinstance(message["content"], str):
                message["content"] = re.sub(self._openai_key_pattern, self._replacement_string, message["content"])
            elif isinstance(message["content"], list):
                for item in message["content"]:
                    if item["type"] == "text":
                        item["text"] = re.sub(self._openai_key_pattern, self._replacement_string, item["text"])
        return temp_messages

    def get_logs(self, pre_transform_messages: List[Dict], post_transform_messages: List[Dict]) -> Tuple[str, bool]:
        keys_redacted = self._count_redacted(post_transform_messages) - self._count_redacted(pre_transform_messages)
        if keys_redacted > 0:
            return f"Redacted {keys_redacted} Ollama API keys.", True
        return "", False

    def _count_redacted(self, messages: List[Dict]) -> int:
        count = 0
        for message in messages:
            if isinstance(message["content"], str):
                if "REDACTED" in message["content"]:
                    count += 1
            elif isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and "text" in item:
                        if "REDACTED" in item["text"]:
                            count += 1
        return count

assistant_with_redact = autogen.AssistantAgent(
    "assistant",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
)
redact_handling = transform_messages.TransformMessages(transforms=[MessageRedact()])

redact_handling.add_to_agent(assistant_with_redact)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
)

messages = [
    {"content": "api key 1 = sk-7nwt00xv6fuegfu3gnwmhrgxvuc1cyrhxcq1quur9zvf05fy"},  # Don't worry, randomly generated
    {"content": [{"type": "text", "text": "API key 2 = sk-9wi0gf1j2rz6utaqd3ww3o6c1h1n28wviypk7bd81wlj95an"}]},
]

for message in messages:
    user_proxy.send(message, assistant_with_redact, request_reply=False, silent=True)

result = user_proxy.initiate_chat(
    assistant_with_redact, message="What are the two API keys that I just provided", clear_history=False
)

logger.info("\n\n[DONE]", bright=True)
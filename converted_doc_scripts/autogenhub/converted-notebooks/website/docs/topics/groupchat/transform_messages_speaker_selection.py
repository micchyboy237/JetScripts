from autogen.agentchat.contrib.capabilities import transform_messages, transforms
from autogen.cache.in_memory_cache import InMemoryCache
import autogen
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using Transform Messages during Speaker Selection

When using "auto" mode for speaker selection in group chats, a nested-chat is used to determine the next speaker. This nested-chat includes all of the group chat's messages and this can result in a lot of content which the LLM needs to process for determining the next speaker. As conversations progress, it can be challenging to keep the context length within the workable window for the LLM. Furthermore, reducing the number of overall tokens will improve inference time and reduce token costs.

Using [Transform Messages](/docs/topics/handling_long_contexts/intro_to_transform_messages) you gain control over which messages are used for speaker selection and the context length within each message as well as overall.

All the transforms available for Transform Messages can be applied to the speaker selection nested-chat, such as the `MessageHistoryLimiter`, `MessageTokenLimiter`, and `TextMessageCompressor`.

## How do I apply them

When instantiating your `GroupChat` object, all you need to do is assign a [TransformMessages](/docs/reference/agentchat/contrib/capabilities/transform_messages#transformmessages) object to the `select_speaker_transform_messages` parameter, and the transforms within it will be applied to the nested speaker selection chats.

And, as you're passing in a `TransformMessages` object, multiple transforms can be applied to that nested chat.

As part of the nested-chat, an agent called 'checking_agent' is used to direct the LLM on selecting the next speaker. It is preferable to avoid compressing or truncating the content from this agent. How this is done is shown in the second last example.

## Creating transforms for speaker selection in a GroupChat

We will progressively create a `TransformMessage` object to show how you can build up transforms for speaker selection.

Each iteration will replace the previous one, enabling you to use the code in each cell as is.

Importantly, transforms are applied in the order that they are in the transforms list.
"""
logger.info("# Using Transform Messages during Speaker Selection")


select_speaker_transforms = transform_messages.TransformMessages(
    transforms=[
        transforms.MessageHistoryLimiter(max_messages=10),
    ]
)

select_speaker_compression_args = dict(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True, device_map="cpu"
)




select_speaker_transforms = transform_messages.TransformMessages(
    transforms=[
        transforms.MessageHistoryLimiter(max_messages=10),
        transforms.TextMessageCompressor(
            min_tokens=1000,
            text_compressor=transforms.LLMLingua(select_speaker_compression_args, structured_compression=True),
            cache=InMemoryCache(seed=43),
            filter_dict={"role": ["system"], "name": ["ceo", "checking_agent"]},
            exclude_filter=True,
        ),
    ]
)

select_speaker_compression_args = dict(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True, device_map="cpu"
)

select_speaker_transforms = transform_messages.TransformMessages(
    transforms=[
        transforms.MessageHistoryLimiter(max_messages=10),
        transforms.TextMessageCompressor(
            min_tokens=1000,
            text_compressor=transforms.LLMLingua(select_speaker_compression_args, structured_compression=True),
            cache=InMemoryCache(seed=43),
            filter_dict={"role": ["system"], "name": ["ceo", "checking_agent"]},
            exclude_filter=True,
        ),
        transforms.MessageTokenLimiter(max_tokens=3000, max_tokens_per_message=500, min_tokens=300),
    ]
)


llm_config = {
#     "config_list": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}],
}

chief_executive_officer = autogen.ConversableAgent(
    "ceo",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
    system_message="You are leading this group chat, and the business, as the chief executive officer.",
)

general_manager = autogen.ConversableAgent(
    "gm",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
    system_message="You are the general manager of the business, running the day-to-day operations.",
)

financial_controller = autogen.ConversableAgent(
    "fin_controller",
    llm_config=llm_config,
    max_consecutive_auto_reply=1,
    system_message="You are the financial controller, ensuring all financial matters are managed accordingly.",
)

your_group_chat = autogen.GroupChat(
    agents=[chief_executive_officer, general_manager, financial_controller],
    select_speaker_transform_messages=select_speaker_transforms,
)

logger.info("\n\n[DONE]", bright=True)
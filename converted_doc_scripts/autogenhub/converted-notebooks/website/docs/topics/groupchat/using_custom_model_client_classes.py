from autogen.agentchat import ConversableAgent, GroupChat, GroupChatManager
from types import SimpleNamespace
import random

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Using Custom Model Client classes with Auto Speaker Selection

````{=mdx}
:::tip
This documentation only applies when using the 'auto' speaker selection method for GroupChat **and** your GroupChatManager is using a Custom Model Client class.

You don't need to change your GroupChat if either of these are the case:
- You are using a different speaker selection method, such as 'manual', 'random', 'round_robin', or a Callable
- Your GroupChatManager doesn't use a Custom Model Client class
:::
````

During a group chat using the `auto` speaker selection method, an inner conversation between two agents is created to determine the next speaker after each turn. One of the speakers will take the `llm_config` from the `GroupChatManager` (the other inner agent doesn't use an `llm_config`).

If the configuration for the GroupChatManager is using a Custom Model Client Class this is not propagated through to the inner conversation.

So, you can control the configuration that the inner conversation agent uses by setting two properties on GroupChat:

- **select_speaker_auto_llm_config**: Set this to your llm_config with the custom model client
- **select_speaker_auto_model_client_cls**: Set this to the class of your custom model client

This control enables you to register the custom model client class for, or assign a completely different `llm_config` to, the inner conversation agent.

See a simple example below.

### Imports
"""
logger.info("# Using Custom Model Client classes with Auto Speaker Selection")


"""
### Sample Custom Model Client class
The class below is an example of a custom model client class that always returns the name `Alexandra`.
"""
logger.info("### Sample Custom Model Client class")



class MyCustomModelClient:
    def __init__(self, config, **kwargs):
        logger.debug(f"CustomModelClient config: {config}")

    def create(self, params):
        num_of_responses = params.get("n", 1)

        response = SimpleNamespace()
        response.choices = []
        response.model = "anything"

        agent_names = ["Alexandra", "Mark", "Elizabeth"]
        random_index = random.randint(0, 2)

        for _ in range(num_of_responses):
            text = f"Randomly choosing... {agent_names[random_index]}"
            choice = SimpleNamespace()
            choice.message = SimpleNamespace()
            choice.message.content = text
            choice.message.function_call = None
            response.choices.append(choice)
        return response

    def message_retrieval(self, response):
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response) -> float:
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        return {}

"""
### GroupChat with Custom Model Client class
Here we create `llm_config` that will use an actual LLM, then we create `custom_llm_config` that uses the custom model client class that we specified earlier.

We add a few agents, all using the LLM-based configuration.
"""
logger.info("### GroupChat with Custom Model Client class")

llm_config = {
    "config_list": [
        {
            "api_type": "ollama",
            "model": "llama3.1:8b",
        }
    ]
}

custom_llm_config = {
    "config_list": [
        {
            "model_client_cls": "MyCustomModelClient",
        }
    ]
}

mark = ConversableAgent(
    name="Mark",
    system_message="You are a customer who likes asking questions about accounting.",
    description="Customer who needs accounting help.",
    llm_config=llm_config,
)

alexandra = ConversableAgent(
    name="Alexandra",
    system_message="You are an accountant who provides detailed responses about accounting.",
    description="Accountant who loves to talk about accounting!",
    llm_config=llm_config,
)

elizabeth = ConversableAgent(
    name="Elizabeth",
    system_message="You are a head accountant who checks the answers of other accountants. Finish your response with the word 'BAZINGA'.",
    description="Head accountants, checks answers from accountants for validity.",
    llm_config=llm_config,
)

"""
Now, we assign the `custom_llm_config` (which uses the custom model client class) and the custom model client class, `MyCustomModelClient`, to the GroupChat so the inner conversation will use it.
"""
logger.info("Now, we assign the `custom_llm_config` (which uses the custom model client class) and the custom model client class, `MyCustomModelClient`, to the GroupChat so the inner conversation will use it.")

gc = GroupChat(
    agents=[mark, alexandra, elizabeth],
    speaker_selection_method="auto",
    allow_repeat_speaker=False,
    select_speaker_auto_verbose=True,
    select_speaker_auto_llm_config=custom_llm_config,
    select_speaker_auto_model_client_cls=MyCustomModelClient,
    max_round=5,
    messages=[],
)

"""
With that setup, we create the GroupChatManager, which will use the LLM-based config. So, the custom model client class will only be used for the inner, select speaker, agent of the GroupChat.
"""
logger.info("With that setup, we create the GroupChatManager, which will use the LLM-based config. So, the custom model client class will only be used for the inner, select speaker, agent of the GroupChat.")

gcm = GroupChatManager(
    groupchat=gc,
    name="moderator",
    system_message="You are moderating a chat between a customer, an accountant, and a head accountant. The customer asks a question, the accountant answers it, and the head accountant then validates it.",
    is_termination_msg=lambda msg: "BAZINGA" in msg["content"].lower(),
    llm_config=llm_config,
)

result = gcm.initiate_chat(
    recipient=mark,
    message="Mark, ask us an accounting question. Alexandra and Elizabeth will help you out.",
    summary_method="last_msg",
)

"""
We can see that the inner `speaker_selection_agent` was returning random names for the next agent, highlighting how we can control the configuration that that inner agent used for `auto` speaker selection in group chats.
"""
logger.info("We can see that the inner `speaker_selection_agent` was returning random names for the next agent, highlighting how we can control the configuration that that inner agent used for `auto` speaker selection in group chats.")

logger.info("\n\n[DONE]", bright=True)
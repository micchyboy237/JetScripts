from jet.transformers.formatters import format_json
from deepeval import evaluate
from deepeval.dataset import ConversationalGolden
from deepeval.metrics import TurnRelevancyMetric
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Turn
from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
---
id: conversation-simulator
title: Conversation Simulator
sidebar_label: Conversation Simulator
---

<head>
  <link
    rel="canonical"
    href="https://deepeval.com/docs/conversation-simulator"
  />
</head>

`deepeval`'s `ConversationSimulator` allows you to simulate full conversations between a fake user and your chatbot, unlike the [synthesizer](/docs/synthesizer-introduction) which generates regular goldens representing single, atomic LLM interactions.
"""
logger.info("id: conversation-simulator")


conversation_golden = ConversationalGolden(
    scenario="Andy Byron wants to purchase a VIP ticket to a cold play concert.",
    expected_outcome="Successful purchase of a ticket.",
    user_description="Andy Byron is the CEO of Astronomer.",
)

async def chatbot_callback(input):
    return Turn(role="assistant", content=f"Chatbot response to: {input}")

simulator = ConversationSimulator(model_callback=chatbot_callback)
conversational_test_cases = simulator.simulate(conversational_goldens=[conversation_golden])
logger.debug(conversational_test_cases)

"""
The `ConversationSimulator` uses the scenario and user description from a `ConversationalGolden` to simulate back-and-forth exchanges with your chatbot. The resulting dialogue is used to create `ConversationalTestCase`s for evaluation using `deepeval`'s multi-turn metrics.

## Create Your First Simulator

To create a `ConversationSimulator`, you'll need to define a callback that wraps around your LLM chatbot.
"""
logger.info("## Create Your First Simulator")


async def model_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    return Turn(role="assistant", content=f"I don't know how to answer this: {input}")

simulator = ConversationSimulator(model_callback=model_callback)

"""
There are **ONE** mandatory and **FOUR** optional parameters when creating a `ConversationSimulator`:

- `model_callback`: a callback that wraps around your conversational agent.
- [Optional] `simulator_model`: a string specifying which of Ollama's GPT models to use for generation, **OR** [any custom LLM model](/docs/metrics-introduction#using-a-custom-llm) of type `DeepEvalBaseLLM`. Defaulted to `gpt-4.1`.
- [Optional] `opening_message`: a string that specifies your LLM chatbot's opening message. You should only provide this **IF** your chatbot is designed to talk before a user does. Defaulted to `None`.
- [Optional] `async_mode`: a boolean which when set to `True`, enables **concurrent simulation of conversations**. Defaulted to `True`.
- [Optional] `max_concurrent`: an integer that determines the maximum number of conversations that can be generated in parallel at any point in time. You can decrease this value if you're running into rate limit errors. Defaulted to `100`.

### Model callback

Only the `input` argument is required when defining your `model_callback`, but you may also define these optional arguments:

- [Optional] `turns`: a list of `Turn`s, which include the role and content of each message in the conversation.
- [Optional] `thread_id`: a unique identifier for each conversation.

While turns captures the dialogue context for each turn, some applications must persist additional state across turns — for example, when invoking external APIs or tracking user-specific data. In these cases, you'll want to take advantage of the `thread_id`.
"""
logger.info("### Model callback")


async def model_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:

    logger.debug(turns)
    logger.debug(thread_id)

    res = await your_llm_app(input, turns, thread_id)
    logger.success(format_json(res))
    return Turn(role="assistant", content=res)

"""
## Simulate A Conversation

To simulate your first conversation, simply pass in a list of `ConversationalGolden`s to the `simulate` method:
"""
logger.info("## Simulate A Conversation")

...

conversation_golden = ConversationalGolden(
    scenario="Andy Byron wants to purchase a VIP ticket to a cold play concert.",
    expected_outcome="Successful purchase of a ticket.",
    user_description="Andy Byron is the CEO of Astronomer.",
)
conversational_test_cases = simulator.simulate(conversational_goldens=[conversation_golden])

"""
There are **ONE** mandatory and **ONE** optional parameter when calling the `simulate` method:

- `conversational_goldens`: a list of `ConversationalGolden`s that specify the scenario and user description.
- [Optional] `max_user_simulations`: an integer that specifies the maximum number of user-assistant message cycles to simulate per conversation. Defaulted to `10`.

A simulation ends either when the converaiton achieves the expected outcome outlined in a `ConversationalGolden`, or when the `max_user_simulations` has been reached.

:::tip
You can also generate conversations from existing turns. Simply populate your `ConversationalGolden` with a list of initial `Turn`s, and the simulator will continue the conversation.
:::

## Evaluate Simulated Turns

The `simulate` function returns a list of `ConversationalTestCase`s, which can be used to evaluate your LLM chatbot using `deepeval`'s conversational metrics. Use simulated conversations to run [end-to-end](/docs/evaluation-end-to-end-llm-evals) evaluations:
"""
logger.info("## Evaluate Simulated Turns")

...

evaluate(test_cases=conversational_test_cases, metrics=[TurnRelevancyMetric()])

logger.info("\n\n[DONE]", bright=True)
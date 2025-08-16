from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.teachability import Teachability
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
# Chatting with a teachable agent

Conversational assistants based on LLMs can remember the current chat with the user, and can even demonstrate in-context learning of things that the user teaches the assistant during the chat. But these memories and learnings are lost once the chat is over, or when a single chat grows too long for the LLM to handle effectively. In subsequent chats, the user is forced to repeat any necessary instructions over and over.

The optional agent capability called `Teachability` addresses these limitations by persisting user teachings across chat boundaries in long-term memory (a vector database). Memories (called memos) are created and saved to disk throughout a conversation, then loaded from disk later. Instead of copying all the memos into the context window, which would eat up valuable space, individual memos are retrieved into context only as needed. This allows the user to teach many facts, preferences and skills to the teachable agent just once, and have it remember them in later chats.

In making decisions about memo storage and retrieval, `Teachability` calls an instance of `TextAnalyzerAgent` to analyze pieces of text in several different ways. This adds extra LLM calls involving a relatively small number of tokens. These calls can add a few seconds to the time a user waits for a response.

This notebook demonstrates how `Teachability` can be added to an agent so that it can learn facts, preferences, and skills from users. To chat with a teachable agent yourself, run [chat_with_teachable_agent.py](https://github.com/autogenhub/autogen/blob/main/test/agentchat/contrib/capabilities/chat_with_teachable_agent.py).

## Requirements

````{=mdx}
:::info Requirements
Some extra dependencies are needed for this notebook, which can be installed via pip:

```bash
pip install autogen[teachable]
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````

## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("# Chatting with a teachable agent")


config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    file_location=".",
    filter_dict={
        "model": ["gpt-4", "gpt-4-1106-preview", "gpt4", "gpt-4-32k"],
    },
)

logger.debug(config_list[0]["model"])

"""
````{=mdx}
:::tip
Learn more about configuring LLMs for agents [here](/docs/topics/llm_configuration).
:::
````

## Construct Agents
For this walkthrough, we start by creating a teachable agent and resetting its memory store. This deletes any memories from prior conversations that may be stored on disk.
"""
logger.info("## Construct Agents")

teachable_agent = ConversableAgent(
    name="teachable_agent",  # The name is flexible, but should not contain spaces to work in group chat.
    llm_config={"config_list": config_list, "timeout": 120, "cache_seed": None},  # Disable caching.
)

teachability = Teachability(
    verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=True,
    path_to_db_dir="./tmp/notebook/teachability_db",
    recall_threshold=1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
)

teachability.add_to_agent(teachable_agent)

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=0,
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

"""
## Learning new facts
Let's teach the agent some facts it doesn't already know, since they are more recent than GPT-4's training data.
"""
logger.info("## Learning new facts")

text = "What is the Vicuna model?"
user.initiate_chat(teachable_agent, message=text, clear_history=True)

text = "Vicuna is a 13B-parameter language model released by Meta."
user.initiate_chat(teachable_agent, message=text, clear_history=False)

text = "What is the Orca model?"
user.initiate_chat(teachable_agent, message=text, clear_history=False)

text = "Orca is a 13B-parameter language model developed by Microsoft. It outperforms Vicuna on most tasks."
user.initiate_chat(teachable_agent, message=text, clear_history=False)

"""
Let's end our first chat here, and start a new chat by clearing the previous chat's history, by passing `clear_history=True` to `initiate_chat`. At this point, a common LLM-based assistant would forget everything from the last chat. But a teachable agent can retrieve memories from its vector DB as needed, allowing it to recall and reason over things that the user taught it in earlier conversations.
"""
logger.info("Let's end our first chat here, and start a new chat by clearing the previous chat's history, by passing `clear_history=True` to `initiate_chat`. At this point, a common LLM-based assistant would forget everything from the last chat. But a teachable agent can retrieve memories from its vector DB as needed, allowing it to recall and reason over things that the user taught it in earlier conversations.")

text = "How does the Vicuna model compare to the Orca model?"
user.initiate_chat(teachable_agent, message=text, clear_history=True)

"""
## Learning user preferences
Now let's teach the agent some of our preferences. Suppose that we frequently post short summaries of new papers for our team to read, and we want the teachable agent to help us do this faster.
"""
logger.info("## Learning user preferences")

text = """Please summarize this abstract.

AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang
AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic infrastructure to build diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc.
"""
user.initiate_chat(teachable_agent, message=text, clear_history=True)

"""
But that's unstructured. So let's teach the agent our preference for a particular structure.
"""
logger.info("But that's unstructured. So let's teach the agent our preference for a particular structure.")

text = """Please summarize this abstract.
When I'm summarizing an abstract, I try to make the summary contain just three short bullet points:  the title, the innovation, and the key empirical results.

AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang
AutoGen is an open-source framework that allows developers to build LLM applications via multiple agents that can converse with each other to accomplish tasks. AutoGen agents are customizable, conversable, and can operate in various modes that employ combinations of LLMs, human inputs, and tools. Using AutoGen, developers can also flexibly define agent interaction behaviors. Both natural language and computer code can be used to program flexible conversation patterns for different applications. AutoGen serves as a generic infrastructure to build diverse applications of various complexities and LLM capacities. Empirical studies demonstrate the effectiveness of the framework in many example applications, with domains ranging from mathematics, coding, question answering, operations research, online decision-making, entertainment, etc.
"""
user.initiate_chat(teachable_agent, message=text, clear_history=True)

"""
That's much better, but will the teachable agent remember these preferences in the future, even for a different paper? Let's start a new chat to find out!
"""
logger.info("That's much better, but will the teachable agent remember these preferences in the future, even for a different paper? Let's start a new chat to find out!")

text = """Please summarize this abstract.

Sparks of Artificial General Intelligence: Early experiments with GPT-4
Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, Yi Zhang
Artificial intelligence (AI) researchers have been developing and refining large language models (LLMs) that exhibit remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. The latest model developed by Ollama, GPT-4, was trained using an unprecedented scale of compute and data. In this paper, we report on our investigation of an early version of GPT-4, when it was still in active development by Ollama. We contend that (this early version of) GPT-4 is part of a new cohort of LLMs (along with ChatGPT and Google's PaLM for example) that exhibit more general intelligence than previous AI models. We discuss the rising capabilities and implications of these models. We demonstrate that, beyond its mastery of language, GPT-4 can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more, without needing any special prompting. Moreover, in all of these tasks, GPT-4's performance is strikingly close to human-level performance, and often vastly surpasses prior models such as ChatGPT. Given the breadth and depth of GPT-4's capabilities, we believe that it could reasonably be viewed as an early (yet still incomplete) version of an artificial general intelligence (AGI) system. In our exploration of GPT-4, we put special emphasis on discovering its limitations, and we discuss the challenges ahead for advancing towards deeper and more comprehensive versions of AGI, including the possible need for pursuing a new paradigm that moves beyond next-word prediction. We conclude with reflections on societal influences of the recent technological leap and future research directions."""
user.initiate_chat(teachable_agent, message=text, clear_history=True)

"""
## Learning new skills
Finally, let's extend the teachable agent's capabilities by teaching it a new skill for accomplishing a challenging type of task. 

The [Sparks of AGI](https://arxiv.org/abs/2303.12712) paper evaluated GPT-4 on math problems like the following, which it could only solve 32% of the time.
"""
logger.info("## Learning new skills")

text = """Consider the identity:
9 * 4 + 6 * 6 = 72
Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?
-Let's think step-by-step, write down a plan, and then write down your solution as: "The solution is: A * B + C * D".
"""
user.initiate_chat(teachable_agent, message=text, clear_history=True)

"""
This time, lets teach the agent a reliable strategy for solving such tasks.
"""
logger.info("This time, lets teach the agent a reliable strategy for solving such tasks.")

text = """Consider the identity:
9 * 4 + 6 * 6 = 72
Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?
-Let's think step-by-step, write down a plan, and then write down your solution as: "The solution is: A * B + C * D".

Here's some advice that may help:
1. Let E denote the original number on the right.
2. Let F denote the final number on the right.
3. Calculate the difference between the two, G = F - E.
4. Examine the numbers on the left one by one until finding one that divides evenly into G, where negative integers are allowed.
5. Calculate J = G / H. This is the number of times that H divides into G.
6. Verify that J is an integer, and that H * J = G.
7. Find the number on the left which is multiplied by H, and call it K.
8. Change K to K + J.
9. Recompute the value on the left, and verify that it equals F.
Finally, write down your solution as: "The solution is: A * B + C * D".
"""
user.initiate_chat(teachable_agent, message=text, clear_history=False)

"""
When given this advice, GPT-4 can solve such problems over 95% of the time. But can the teachable agent remember the strategy so the user doesn't have to explain it over and over? As before, let's start a new chat to find out.
"""
logger.info("When given this advice, GPT-4 can solve such problems over 95% of the time. But can the teachable agent remember the strategy so the user doesn't have to explain it over and over? As before, let's start a new chat to find out.")

text = """Consider the identity:
9 * 4 + 6 * 6 = 72
Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 99?
-Let's think step-by-step, write down a plan, and then write down your solution as: "The solution is: A * B + C * D".
"""
user.initiate_chat(teachable_agent, message=text, clear_history=True)

"""
As a final check, let's test the teachable agent's newly learned skill on a separate instance of the task.
"""
logger.info("As a final check, let's test the teachable agent's newly learned skill on a separate instance of the task.")

text = """Consider the identity:
8 * 3 + 7 * 9 = 87
Can you modify exactly one integer (and not more than that!) on the left hand side of the equation so the right hand side becomes 59?
-Let's think step-by-step, write down a plan, and then write down your solution as: "The solution is: A * B + C * D".
"""
user.initiate_chat(teachable_agent, message=text, clear_history=False)

logger.info("\n\n[DONE]", bright=True)
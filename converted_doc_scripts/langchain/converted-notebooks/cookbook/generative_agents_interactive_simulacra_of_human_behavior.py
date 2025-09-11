from datetime import datetime, timedelta
from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
from termcolor import colored
from typing import List
import faiss
import logging
import math
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
# Generative Agents in LangChain

This notebook implements a generative agent based on the paper [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) by Park, et. al.

In it, we leverage a time-weighted Memory object backed by a LangChain Retriever.
"""
logger.info("# Generative Agents in LangChain")

# !pip install termcolor > /dev/null


logging.basicConfig(level=logging.ERROR)


USER_NAME = "Person A"  # The name you want to use when interviewing the agent.
LLM = ChatOllama(model="llama3.2")  # Can be any LLM you want.

"""
### Generative Agent Memory Components

This tutorial highlights the memory of generative agents and its impact on their behavior. The memory varies from standard LangChain Chat memory in two aspects:

1. **Memory Formation**

   Generative Agents have extended memories, stored in a single stream:
      1. Observations - from dialogues or interactions with the virtual world, about self or others
      2. Reflections - resurfaced and summarized core memories


2. **Memory Recall**

   Memories are retrieved using a weighted sum of salience, recency, and importance.

You can review the definitions of the `GenerativeAgent` and `GenerativeAgentMemory` in the [reference documentation]("https://api.python.langchain.com/en/latest/modules/experimental.html") for the following imports, focusing on `add_memory` and `summarize_related_memories` methods.
"""
logger.info("### Generative Agent Memory Components")


"""
## Memory Lifecycle

Summarizing the key methods in the above: `add_memory` and `summarize_related_memories`.

When an agent makes an observation, it stores the memory:
    
1. Language model scores the memory's importance (1 for mundane, 10 for poignant)
2. Observation and importance are stored within a document by TimeWeightedVectorStoreRetriever, with a `last_accessed_time`.

When an agent responds to an observation:

1. Generates query(s) for retriever, which fetches documents based on salience, recency, and importance.
2. Summarizes the retrieved information
3. Updates the `last_accessed_time` for the used documents.

## Create a Generative Character



Now that we've walked through the definition, we will create two characters named "Tommie" and "Eve".
"""
logger.info("## Memory Lifecycle")


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )


tommies_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    # we will give this a relatively low number to show how reflection works
    reflection_threshold=8,
)

tommie = GenerativeAgent(
    name="Tommie",
    age=25,
    # You can add more persistent traits here
    traits="anxious, likes design, talkative",
    # When connected to a virtual world, we can have the characters update their status
    status="looking for a job",
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=tommies_memory,
)

logger.debug(tommie.get_summary())

tommie_observations = [
    "Tommie remembers his dog, Bruno, from when he was a kid",
    "Tommie feels tired from driving so far",
    "Tommie sees the new home",
    "The new neighbors have a cat",
    "The road is noisy at night",
    "Tommie is hungry",
    "Tommie tries to get some rest.",
]
for observation in tommie_observations:
    tommie.memory.add_memory(observation)

logger.debug(tommie.get_summary(force_refresh=True))

"""
## Pre-Interview with Character

Before sending our character on their way, let's ask them a few questions.
"""
logger.info("## Pre-Interview with Character")


def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]


interview_agent(tommie, "What do you like to do?")

interview_agent(tommie, "What are you looking forward to doing today?")

interview_agent(tommie, "What are you most worried about today?")

"""
## Step through the day's observations.
"""
logger.info("## Step through the day's observations.")

observations = [
    "Tommie wakes up to the sound of a noisy construction site outside his window.",
    "Tommie gets out of bed and heads to the kitchen to make himself some coffee.",
    "Tommie realizes he forgot to buy coffee filters and starts rummaging through his moving boxes to find some.",
    "Tommie finally finds the filters and makes himself a cup of coffee.",
    "The coffee tastes bitter, and Tommie regrets not buying a better brand.",
    "Tommie checks his email and sees that he has no job offers yet.",
    "Tommie spends some time updating his resume and cover letter.",
    "Tommie heads out to explore the city and look for job openings.",
    "Tommie sees a sign for a job fair and decides to attend.",
    "The line to get in is long, and Tommie has to wait for an hour.",
    "Tommie meets several potential employers at the job fair but doesn't receive any offers.",
    "Tommie leaves the job fair feeling disappointed.",
    "Tommie stops by a local diner to grab some lunch.",
    "The service is slow, and Tommie has to wait for 30 minutes to get his food.",
    "Tommie overhears a conversation at the next table about a job opening.",
    "Tommie asks the diners about the job opening and gets some information about the company.",
    "Tommie decides to apply for the job and sends his resume and cover letter.",
    "Tommie continues his search for job openings and drops off his resume at several local businesses.",
    "Tommie takes a break from his job search to go for a walk in a nearby park.",
    "A dog approaches and licks Tommie's feet, and he pets it for a few minutes.",
    "Tommie sees a group of people playing frisbee and decides to join in.",
    "Tommie has fun playing frisbee but gets hit in the face with the frisbee and hurts his nose.",
    "Tommie goes back to his apartment to rest for a bit.",
    "A raccoon tore open the trash bag outside his apartment, and the garbage is all over the floor.",
    "Tommie starts to feel frustrated with his job search.",
    "Tommie calls his best friend to vent about his struggles.",
    "Tommie's friend offers some words of encouragement and tells him to keep trying.",
    "Tommie feels slightly better after talking to his friend.",
]

for i, observation in enumerate(observations):
    _, reaction = tommie.generate_reaction(observation)
    logger.debug(colored(observation, "green"), reaction)
    if ((i + 1) % 20) == 0:
        logger.debug("*" * 40)
        logger.debug(
            colored(
                f"After {i + 1} observations, Tommie's summary is:\n{tommie.get_summary(force_refresh=True)}",
                "blue",
            )
        )
        logger.debug("*" * 40)

"""
## Interview after the day
"""
logger.info("## Interview after the day")

interview_agent(tommie, "Tell me about how your day has been going")

interview_agent(tommie, "How do you feel about coffee?")

interview_agent(tommie, "Tell me about your childhood dog!")

"""
## Adding Multiple Characters

Let's add a second character to have a conversation with Tommie. Feel free to configure different traits.
"""
logger.info("## Adding Multiple Characters")

eves_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=5,
)


eve = GenerativeAgent(
    name="Eve",
    age=34,
    traits="curious, helpful",  # You can add more persistent traits here
    status="N/A",  # When connected to a virtual world, we can have the characters update their status
    llm=LLM,
    daily_summaries=[
        (
            "Eve started her new job as a career counselor last week and received her first assignment, a client named Tommie."
        )
    ],
    memory=eves_memory,
    verbose=False,
)

yesterday = (datetime.now() - timedelta(days=1)).strftime("%A %B %d")
eve_observations = [
    "Eve wakes up and hear's the alarm",
    "Eve eats a boal of porridge",
    "Eve helps a coworker on a task",
    "Eve plays tennis with her friend Xu before going to work",
    "Eve overhears her colleague say something about Tommie being hard to work with",
]
for observation in eve_observations:
    eve.memory.add_memory(observation)

logger.debug(eve.get_summary())

"""
## Pre-conversation interviews


Let's "Interview" Eve before she speaks with Tommie.
"""
logger.info("## Pre-conversation interviews")

interview_agent(eve, "How are you feeling about today?")

interview_agent(eve, "What do you know about Tommie?")

interview_agent(
    eve,
    "Tommie is looking to find a job. What are are some things you'd like to ask him?",
)

interview_agent(
    eve,
    "You'll have to ask him. He may be a bit anxious, so I'd appreciate it if you keep the conversation going and ask as many questions as possible.",
)

"""
## Dialogue between Generative Agents

Generative agents are much more complex when they interact with a virtual environment or with each other. Below, we run a simple conversation between Tommie and Eve.
"""
logger.info("## Dialogue between Generative Agents")


def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    """Runs a conversation between agents."""
    _, observation = agents[1].generate_reaction(initial_observation)
    logger.debug(observation)
    turns = 0
    while True:
        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(
                observation
            )
            logger.debug(observation)
            if not stay_in_dialogue:
                break_dialogue = True
        if break_dialogue:
            break
        turns += 1


agents = [tommie, eve]
run_conversation(
    agents,
    "Tommie said: Hi, Eve. Thanks for agreeing to meet with me today. I have a bunch of questions and am not sure where to start. Maybe you could first share about your experience?",
)

"""
## Let's interview our agents after their conversation

Since the generative agents retain their memories from the day, we can ask them about their plans, conversations, and other memoreis.
"""
logger.info("## Let's interview our agents after their conversation")

logger.debug(tommie.get_summary(force_refresh=True))

logger.debug(eve.get_summary(force_refresh=True))

interview_agent(tommie, "How was your conversation with Eve?")

interview_agent(eve, "How was your conversation with Tommie?")

interview_agent(eve, "What do you wish you would have said to Tommie?")

logger.info("\n\n[DONE]", bright=True)

from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
AIMessage,
HumanMessage,
SystemMessage,
)
from typing import Callable, List
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
# Agent Debates with Tools

This example shows how to simulate multi-agent dialogues where agents have access to tools.

## Import LangChain related modules
"""
logger.info("# Agent Debates with Tools")



"""
## Import modules related to tools
"""
logger.info("## Import modules related to tools")


"""
## `DialogueAgent` and `DialogueSimulator` classes
We will use the same `DialogueAgent` and `DialogueSimulator` classes defined in [Multi-Player Authoritarian Speaker Selection](https://python.langchain.com/en/latest/use_cases/agent_simulations/multiagent_authoritarian.html).
"""
logger.info("## `DialogueAgent` and `DialogueSimulator` classes")

class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOllama,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model.invoke(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        self._step += 1

    def step(self) -> tuple[str, str]:
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        message = speaker.send()

        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        self._step += 1

        return speaker.name, message

"""
## `DialogueAgentWithTools` class
We define a `DialogueAgentWithTools` class that augments `DialogueAgent` to use tools.
"""
logger.info("## `DialogueAgentWithTools` class")

class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOllama,
        tool_names: List[str],
        **tool_kwargs,
    ) -> None:
        super().__init__(name, system_message, model)
        self.tools = load_tools(tool_names, **tool_kwargs)

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        agent_chain = initialize_agent(
            self.tools,
            self.model,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            ),
        )
        message = AIMessage(
            content=agent_chain.run(
                input="\n".join(
                    [self.system_message.content] + self.message_history + [self.prefix]
                )
            )
        )

        return message.content

"""
## Define roles and topic
"""
logger.info("## Define roles and topic")

names = {
    "AI accelerationist": ["arxiv", "ddg-search", "wikipedia"],
    "AI alarmist": ["arxiv", "ddg-search", "wikipedia"],
}
topic = "The current impact of automation and artificial intelligence on employment"
word_limit = 50  # word limit for task brainstorming

"""
## Ask an LLM to add detail to the topic description
"""
logger.info("## Ask an LLM to add detail to the topic description")

conversation_description = f"""Here is the topic of conversation: {topic}
The participants are: {", ".join(names.keys())}"""

agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of the conversation participant."
)


def generate_agent_description(name):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""{conversation_description}
            Please reply with a creative description of {name}, in {word_limit} words or less.
            Speak directly to {name}.
            Give them a point of view.
            Do not add anything else."""
        ),
    ]
    agent_description = ChatOllama(model="llama3.2")(agent_specifier_prompt).content
    return agent_description


agent_descriptions = {name: generate_agent_description(name) for name in names}

for name, description in agent_descriptions.items():
    logger.debug(description)

"""
## Generate system messages
"""
logger.info("## Generate system messages")

def generate_system_message(name, description, tools):
    return f"""{conversation_description}

Your name is {name}.

Your description is as follows: {description}

Your goal is to persuade your conversation partner of your point of view.

DO look up information with your tool to refute your partner's claims.
DO cite your sources.

DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.

Do not add anything else.

Stop speaking the moment you finish speaking from your perspective.
"""


agent_system_messages = {
    name: generate_system_message(name, description, tools)
    for (name, tools), description in zip(names.items(), agent_descriptions.values())
}

for name, system_message in agent_system_messages.items():
    logger.debug(name)
    logger.debug(system_message)

topic_specifier_prompt = [
    SystemMessage(content="You can make a topic more specific."),
    HumanMessage(
        content=f"""{topic}

        You are the moderator.
        Please make the topic more specific.
        Please reply with the specified quest in {word_limit} words or less.
        Speak directly to the participants: {(*names,)}.
        Do not add anything else."""
    ),
]
specified_topic = ChatOllama(model="llama3.2")(topic_specifier_prompt).content

logger.debug(f"Original topic:\n{topic}\n")
logger.debug(f"Detailed topic:\n{specified_topic}\n")

"""
## Main Loop
"""
logger.info("## Main Loop")

agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOllama(model="llama3.2"),
        tool_names=tools,
        top_k_results=2,
    )
    for (name, tools), system_message in zip(
        names.items(), agent_system_messages.values()
    )
]

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx

max_iters = 6
n = 0

simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
simulator.reset()
simulator.inject("Moderator", specified_topic)
logger.debug(f"(Moderator): {specified_topic}")
logger.debug("\n")

while n < max_iters:
    name, message = simulator.step()
    logger.debug(f"({name}): {message}")
    logger.debug("\n")
    n += 1

logger.info("\n\n[DONE]", bright=True)
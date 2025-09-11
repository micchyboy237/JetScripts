from jet.logger import logger
from langchain.output_parsers import RegexParser
from langchain.schema import (
HumanMessage,
SystemMessage,
)
import os
import shutil
import tenacity


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
# Simulated Environment: Gymnasium

For many applications of LLM agents, the environment is real (internet, database, REPL, etc). However, we can also define agents to interact in simulated environments like text-based games. This is an example of how to create a simple agent-environment interaction loop with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (formerly [Ollama Gym](https://github.com/ollama/gym)).
"""
logger.info("# Simulated Environment: Gymnasium")

# !pip install gymnasium


"""
## Define the agent
"""
logger.info("## Define the agent")

class GymnasiumAgent:
    @classmethod
    def get_docs(cls, env):
        return env.unwrapped.__doc__

    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.docs = self.get_docs(env)

        self.instructions = """
Your goal is to maximize your return, i.e. the sum of the rewards you receive.
I will give you an observation, reward, terminiation flag, truncation flag, and the return so far, formatted as:

Observation: <observation>
Reward: <reward>
Termination: <termination>
Truncation: <truncation>
Return: <sum_of_rewards>

You will respond with an action, formatted as:

Action: <action>

where you replace <action> with your actual action.
Do nothing else but return the action.
"""
        self.action_parser = RegexParser(
            regex=r"Action: (.*)", output_keys=["action"], default_output_key="action"
        )

        self.message_history = []
        self.ret = 0

    def random_action(self):
        action = self.env.action_space.sample()
        return action

    def reset(self):
        self.message_history = [
            SystemMessage(content=self.docs),
            SystemMessage(content=self.instructions),
        ]

    def observe(self, obs, rew=0, term=False, trunc=False, info=None):
        self.ret += rew

        obs_message = f"""
Observation: {obs}
Reward: {rew}
Termination: {term}
Truncation: {trunc}
Return: {self.ret}
        """
        self.message_history.append(HumanMessage(content=obs_message))
        return obs_message

    def _act(self):
        act_message = self.model.invoke(self.message_history)
        self.message_history.append(act_message)
        action = int(self.action_parser.parse(act_message.content)["action"])
        return action

    def act(self):
        try:
            for attempt in tenacity.Retrying(
                stop=tenacity.stop_after_attempt(2),
                wait=tenacity.wait_none(),  # No waiting time between retries
                retry=tenacity.retry_if_exception_type(ValueError),
                before_sleep=lambda retry_state: logger.debug(
                    f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."
                ),
            ):
                with attempt:
                    action = self._act()
        except tenacity.RetryError:
            action = self.random_action()
        return action

"""
## Initialize the simulated environment and agent
"""
logger.info("## Initialize the simulated environment and agent")

env = gym.make("Blackjack-v1")
agent = GymnasiumAgent(model=ChatOllama(model="llama3.2"), env=env)

"""
## Main loop
"""
logger.info("## Main loop")

observation, info = env.reset()
agent.reset()

obs_message = agent.observe(observation)
logger.debug(obs_message)

while True:
    action = agent.act()
    observation, reward, termination, truncation, info = env.step(action)
    obs_message = agent.observe(observation, reward, termination, truncation, info)
    logger.debug(f"Action: {action}")
    logger.debug(obs_message)

    if termination or truncation:
        logger.debug("break", termination, truncation)
        break
env.close()

logger.info("\n\n[DONE]", bright=True)
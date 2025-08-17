from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from cookbooks.helper.mem0_teachability import Mem0Teachability
from jet.logger import CustomLogger
from mem0 import Memory
import logging
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

# %pip install --upgrade pip
# %pip install mem0ai pyautogen flaml


# os.environ["OPENAI_API_KEY"] = "sk-xxx"



logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

assistant_id = os.environ.get("ASSISTANT_ID", None)

CACHE_SEED = 42  # choose your poison
llm_config = {
#     "config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}],
    "cache_seed": CACHE_SEED,
    "timeout": 120,
    "temperature": 0.0,
}

assistant_config = {"assistant_id": assistant_id}

gpt_assistant = GPTAssistantAgent(
    name="assistant",
    instructions=AssistantAgent.DEFAULT_SYSTEM_MESSAGE,
    llm_config=llm_config,
    assistant_config=assistant_config,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    llm_config=llm_config,
)

user_query = "Write a Python function that reverses a string."
user_proxy.initiate_chat(gpt_assistant, message=user_query)


MEM0_MEMORY_CLIENT = Memory()

USER_ID = "chicory.ai.user"
MEMORY_DATA = """
* Preference for readability: The user prefers code to be explicitly written with clear variable names.
* Preference for comments: The user prefers comments explaining each step.
* Naming convention: The user prefers camelCase for variable names.
* Docstrings: The user prefers functions to have a descriptive docstring.
"""
AGENT_ID = "chicory.ai"

MEM0_MEMORY_CLIENT.add(MEMORY_DATA, user_id=USER_ID)
MEM0_MEMORY_CLIENT.add(MEMORY_DATA, agent_id=AGENT_ID)

"""
## Option 1: 
Using Direct Prompt Injection:
`user memory example`
"""
logger.info("## Option 1:")

relevant_memories = MEM0_MEMORY_CLIENT.search(user_query, user_id=USER_ID, limit=3)
relevant_memories_text = "\n".join(mem["memory"] for mem in relevant_memories)
logger.debug("Relevant memories:")
logger.debug(relevant_memories_text)

prompt = f"{user_query}\n Coding Preferences: \n{relevant_memories_text}"
browse_result = user_proxy.initiate_chat(gpt_assistant, message=prompt)

"""
## Option 2:
Using UserProxyAgent: 
`agent memory example`
"""
logger.info("## Option 2:")

class Mem0ProxyCoderAgent(UserProxyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = MEM0_MEMORY_CLIENT
        self.agent_id = kwargs.get("name")

    def initiate_chat(self, assistant, message):
        agent_memories = self.memory.search(message, agent_id=self.agent_id, limit=3)
        agent_memories_txt = "\n".join(mem["memory"] for mem in agent_memories)
        prompt = f"{message}\n Coding Preferences: \n{str(agent_memories_txt)}"
        response = super().initiate_chat(assistant, message=prompt)
        response_dist = response.__dict__ if not isinstance(response, dict) else response
        MEMORY_DATA = [{"role": "user", "content": message}, {"role": "assistant", "content": response_dist}]
        self.memory.add(MEMORY_DATA, agent_id=self.agent_id)
        return response

mem0_user_proxy = Mem0ProxyCoderAgent(
    name=AGENT_ID,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
)
code_result = mem0_user_proxy.initiate_chat(gpt_assistant, message=user_query)

"""
# Option 3:
Using Teachability:
`agent memory example`
"""
logger.info("# Option 3:")


teachability = Mem0Teachability(
    verbosity=2,  # for visibility of what's happening
    recall_threshold=0.5,
    reset_db=False,  # Use True to force-reset the memo DB, and False to use an existing DB.
    agent_id=AGENT_ID,
    memory_client=MEM0_MEMORY_CLIENT,
)
teachability.add_to_agent(user_proxy)

user_proxy.initiate_chat(gpt_assistant, message=user_query)

logger.info("\n\n[DONE]", bright=True)
from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from jet.logger import CustomLogger
import logging
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# RAG Ollama Assistants in AutoGen

This notebook shows an example of the [`GPTAssistantAgent`](https://github.com/autogenhub/autogen/blob/main/autogen/agentchat/contrib/gpt_assistant_agent.py) with retrieval augmented generation. `GPTAssistantAgent` is an experimental AutoGen agent class that leverages the [Ollama Assistant API](https://platform.openai.com/docs/assistants/overview) for conversational capabilities,  working with
`UserProxyAgent` in AutoGen.
"""
logger.info("# RAG Ollama Assistants in AutoGen")



logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

assistant_id = os.environ.get("ASSISTANT_ID", None)

config_list = config_list_from_json("OAI_CONFIG_LIST")
llm_config = {
    "config_list": config_list,
}
assistant_config = {
    "assistant_id": assistant_id,
    "tools": [{"type": "retrieval"}],
    "file_ids": ["file-AcnBk5PCwAjJMCVO0zLSbzKP"],
}

gpt_assistant = GPTAssistantAgent(
    name="assistant",
    instructions="You are adapt at question answering",
    llm_config=llm_config,
    assistant_config=assistant_config,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="ALWAYS",
)
user_proxy.initiate_chat(gpt_assistant, message="Please explain the code in this file!")

gpt_assistant.delete_assistant()

logger.info("\n\n[DONE]", bright=True)